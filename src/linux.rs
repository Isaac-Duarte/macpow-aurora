//! Linux backend for Apple Silicon Macs running Asahi-based kernels (Aurora, Fedora Asahi).
//!
//! Reads sysfs/procfs instead of IOReport/IOKit. What the kernel does not expose yet
//! (per-block SoC energy: CPU/GPU/ANE/DRAM watts) stays 0; SoC total comes from the
//! SMC "Heatpipe Power" rail and per-process power is that total split by CPU time.

use crate::process_utils::command_output_timeout;
use crate::sma::TimeSma;
use crate::types::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const GPU_STATS: &str = "/sys/devices/platform/soc/2300000000.gpu/agx_stats";
const BAT: &str = "/sys/class/power_supply/macsmc-battery";
const AC: &str = "/sys/class/power_supply/macsmc-ac";
const KBD: &str = "/sys/class/leds/kbd_backlight";

fn read(p: impl AsRef<Path>) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

fn num<T: std::str::FromStr>(p: impl AsRef<Path>) -> Option<T> {
    read(p)?.parse().ok()
}

fn hwmon_dir(name: &str) -> Option<PathBuf> {
    fs::read_dir("/sys/class/hwmon")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| read(p.join("name")).as_deref() == Some(name))
}

/// One labelled hwmon input: (kind prefix like "temp"/"power", label, input path).
struct Sensor {
    kind: String,
    label: String,
    input: PathBuf,
}

fn list_sensors(dir: &Path) -> Vec<Sensor> {
    let Ok(rd) = fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.flatten()
        .filter_map(|e| {
            let f = e.file_name().into_string().ok()?;
            let base = f.strip_suffix("_input")?;
            let kind = base
                .trim_end_matches(|c: char| c.is_ascii_digit())
                .to_string();
            let label = read(dir.join(format!("{base}_label"))).unwrap_or_else(|| base.into());
            Some(Sensor {
                kind,
                label,
                input: e.path(),
            })
        })
        .collect()
}

// ── CPU ──────────────────────────────────────────────────────────────────────

struct Policy {
    dir: PathBuf,
    cpus: Vec<usize>,
    perf: bool,
}

/// cpufreq policies, highest-capacity (P) clusters marked `perf`.
fn read_policies() -> Vec<Policy> {
    let mut pols: Vec<(PathBuf, Vec<usize>, u32)> = fs::read_dir("/sys/devices/system/cpu/cpufreq")
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_name().to_string_lossy().starts_with("policy"))
                .filter_map(|e| {
                    let cpus: Vec<usize> = read(e.path().join("related_cpus"))?
                        .split_whitespace()
                        .filter_map(|c| c.parse().ok())
                        .collect();
                    let cap = num(format!(
                        "/sys/devices/system/cpu/cpu{}/cpu_capacity",
                        cpus.first()?
                    ))
                    .unwrap_or(1024);
                    Some((e.path(), cpus, cap))
                })
                .collect()
        })
        .unwrap_or_default();
    pols.sort_by_key(|p| p.1.first().copied());
    let max_cap = pols.iter().map(|p| p.2).max().unwrap_or(0);
    pols.into_iter()
        .map(|(dir, cpus, cap)| Policy {
            dir,
            cpus,
            perf: cap == max_cap,
        })
        .collect()
}

/// Per-CPU (busy, total) jiffies from /proc/stat, indexed by CPU number.
fn read_cpu_ticks() -> HashMap<usize, (u64, u64)> {
    read("/proc/stat")
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            let id: usize = it.next()?.strip_prefix("cpu")?.parse().ok()?;
            let v: Vec<u64> = it.filter_map(|x| x.parse().ok()).collect();
            let total: u64 = v.iter().take(8).sum();
            let idle = v.get(3)? + v.get(4).unwrap_or(&0);
            Some((id, (total - idle, total)))
        })
        .collect()
}

fn usage_pct(prev: (u64, u64), cur: (u64, u64)) -> f32 {
    let busy = cur.0.saturating_sub(prev.0) as f32;
    let total = cur.1.saturating_sub(prev.1) as f32;
    if total > 0.0 {
        (busy / total * 100.0).clamp(0.0, 100.0)
    } else {
        0.0
    }
}

fn cluster(name: String, cpus: &[usize], prefix: &str) -> CpuCluster {
    CpuCluster {
        name,
        total_w: 0.0,
        cores: (0..cpus.len())
            .map(|i| CpuCore {
                name: format!("{prefix}{i}"),
                watts: 0.0,
            })
            .collect(),
    }
}

fn avg_freq_mhz(pols: &[&Policy]) -> u32 {
    let f: Vec<u32> = pols
        .iter()
        .filter_map(|p| num::<u32>(p.dir.join("scaling_cur_freq")))
        .collect();
    if f.is_empty() {
        0
    } else {
        f.iter().sum::<u32>() / f.len() as u32 / 1000
    }
}

// ── GPU ──────────────────────────────────────────────────────────────────────

/// Parse `key value` lines of the aurora GPU driver's `agx_stats`.
fn read_gpu_stats() -> HashMap<String, u64> {
    read(GPU_STATS)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let (k, v) = l.split_once(' ')?;
            Some((k.to_string(), v.trim().parse().ok()?))
        })
        .collect()
}

// ── Battery / adapter ────────────────────────────────────────────────────────

fn read_battery(sys_power_w: f32) -> BatteryInfo {
    let b = |f: &str| num::<f64>(format!("{BAT}/{f}"));
    let Some(voltage_uv) = b("voltage_now") else {
        return BatteryInfo::default();
    };
    let external = num::<u8>(format!("{AC}/online")) == Some(1);
    let current_ua = b("current_now").unwrap_or(0.0);
    let power_w = (voltage_uv * current_ua).abs() / 1e12;
    let design = b("charge_full_design").unwrap_or(0.0) / 1000.0;
    let full = b("charge_full").unwrap_or(0.0) / 1000.0;
    let secs = if external {
        b("time_to_full_now")
    } else {
        b("time_to_empty_now")
    };
    BatteryInfo {
        present: b("present") != Some(0.0),
        charging: read(format!("{BAT}/status")).as_deref() == Some("Charging"),
        voltage_mv: voltage_uv / 1000.0,
        amperage_ma: current_ua / 1000.0,
        drain_w: if external {
            -power_w
        } else if sys_power_w > 0.0 {
            sys_power_w as f64
        } else {
            power_w
        },
        capacity_wh: b("energy_full").unwrap_or(0.0) / 1e6,
        current_capacity: (b("charge_now").unwrap_or(0.0) / 1000.0) as i64,
        max_capacity: full as i64,
        percent: b("capacity").unwrap_or(0.0),
        time_remaining_min: secs.map_or(-1, |s| (s / 60.0) as i64),
        external_connected: external,
        temperature_c: b("temp").unwrap_or(0.0) / 10.0,
        cycle_count: b("cycle_count").unwrap_or(0.0) as i64,
        design_capacity_mah: design,
        max_capacity_mah: full,
        health_pct: if design > 0.0 {
            full / design * 100.0
        } else {
            100.0
        },
    }
}

// ── Display ──────────────────────────────────────────────────────────────────

/// Built-in panel connector (eDP) directory, if any.
fn edp_connector() -> Option<PathBuf> {
    fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.to_string_lossy().contains("-eDP-"))
}

/// (width, height, diagonal inches) from the connector's mode list and EDID.
fn read_panel_geometry(conn: &Path) -> (u32, u32, f32) {
    let (w, h) = read(conn.join("modes"))
        .and_then(|m| {
            let (w, h) = m.lines().next()?.split_once('x')?;
            Some((w.parse().ok()?, h.trim_end_matches('i').parse().ok()?))
        })
        .unwrap_or((0, 0));
    // EDID bytes 21/22: max image size in cm.
    let diag = fs::read(conn.join("edid"))
        .ok()
        .filter(|e| e.len() > 22)
        .map(|e| ((e[21] as f32).powi(2) + (e[22] as f32).powi(2)).sqrt() / 2.54)
        .map_or(0.0, |d| (d * 10.0).round() / 10.0);
    (w, h, diag)
}

fn read_display(conn: Option<&Path>) -> DisplayInfo {
    let Some(bl) = fs::read_dir("/sys/class/backlight")
        .ok()
        .and_then(|mut rd| rd.next())
        .and_then(|e| e.ok())
        .map(|e| e.path())
    else {
        return DisplayInfo::default();
    };
    let cur: f32 = num(bl.join("actual_brightness"))
        .or_else(|| num(bl.join("brightness")))
        .unwrap_or(0.0);
    let max: f32 = num(bl.join("max_brightness")).unwrap_or(0.0);
    let br = if max > 0.0 { cur / max } else { 0.0 };
    let (width_px, height_px, diagonal_inches) = conn.map_or((0, 0, 0.0), read_panel_geometry);
    DisplayInfo {
        available: max > 0.0,
        brightness_pct: br * 100.0,
        estimated_power_w: br * MAX_DISPLAY_W,
        width_px,
        height_px,
        diagonal_inches,
        ..Default::default()
    }
}

// ── Network ──────────────────────────────────────────────────────────────────

/// iface → (rx_bytes, tx_bytes)
fn read_net_counters() -> HashMap<String, (u64, u64)> {
    read("/proc/net/dev")
        .unwrap_or_default()
        .lines()
        .skip(2)
        .filter_map(|l| {
            let (name, rest) = l.split_once(':')?;
            let v: Vec<u64> = rest
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            Some((name.trim().to_string(), (*v.first()?, *v.get(8)?)))
        })
        .collect()
}

fn net_rate(
    prev: &HashMap<String, (u64, u64)>,
    cur: &HashMap<String, (u64, u64)>,
    dt_s: f64,
    only: Option<&str>,
) -> NetworkInfo {
    let (mut rx, mut tx) = (0u64, 0u64);
    for (k, c) in cur {
        if k == "lo" || only.is_some_and(|o| o != k) {
            continue;
        }
        if let Some(p) = prev.get(k) {
            rx += c.0.saturating_sub(p.0);
            tx += c.1.saturating_sub(p.1);
        }
    }
    NetworkInfo {
        bytes_in_per_sec: rx as f64 / dt_s,
        bytes_out_per_sec: tx as f64 / dt_s,
    }
}

fn find_iface(wireless: bool) -> Option<String> {
    fs::read_dir("/sys/class/net")
        .ok()?
        .flatten()
        .filter(|e| e.path().join("device").exists())
        .find(|e| e.path().join("wireless").exists() == wireless)
        .map(|e| e.file_name().to_string_lossy().into_owned())
}

fn read_ethernet() -> EthernetInfo {
    let Some(iface) = find_iface(false) else {
        return EthernetInfo::default();
    };
    let base = Path::new("/sys/class/net").join(&iface);
    EthernetInfo {
        connected: read(base.join("operstate")).as_deref() == Some("up"),
        link_speed_mbps: num::<i64>(base.join("speed")).unwrap_or(0).max(0) as u32,
        interface_name: iface,
    }
}

fn read_wifi() -> WifiInfo {
    let Some(iface) = find_iface(true) else {
        return WifiInfo::default();
    };
    let mut info = WifiInfo {
        interface_name: iface.clone(),
        ..Default::default()
    };
    let Some(out) =
        command_output_timeout("iw", &["dev", &iface, "link"], Duration::from_millis(500))
    else {
        return info;
    };
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("SSID: ") {
            info.connected = true;
            info.ssid = v.to_string();
        } else if let Some(v) = line.strip_prefix("signal: ") {
            info.rssi_dbm = v
                .split_whitespace()
                .next()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("tx bitrate: ") {
            info.tx_rate_mbps = v
                .split_whitespace()
                .next()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0.0);
            info.phy_mode = ["EHT", "HE", "VHT", "HT"]
                .iter()
                .find(|m| v.contains(*m))
                .map_or("", |m| match *m {
                    "EHT" => "802.11be",
                    "HE" => "802.11ax",
                    "VHT" => "802.11ac",
                    _ => "802.11n",
                })
                .to_string();
        } else if let Some(v) = line.strip_prefix("freq: ") {
            let mhz: f32 = v
                .split_whitespace()
                .next()
                .and_then(|x| x.parse().ok())
                .unwrap_or(0.0);
            info.channel = match mhz as u32 {
                2412..=2484 => "2GHz",
                5000..=5900 => "5GHz",
                5925..=7125 => "6GHz",
                _ => "",
            }
            .to_string();
        }
    }
    if info.connected {
        // Same model as the macOS backend: ~0.1 W idle, up to ~0.8 W at weak signal.
        let q = ((info.rssi_dbm + 100).clamp(0, 60)) as f32 / 60.0;
        info.estimated_power_w = 0.1 + (1.0 - q) * 0.7;
    }
    info
}

// ── Disk ─────────────────────────────────────────────────────────────────────

/// Total (read_bytes, write_bytes) over whole NVMe disks.
fn read_disk_counters() -> (u64, u64) {
    read("/proc/diskstats")
        .unwrap_or_default()
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let name = f.get(2)?;
            (name.starts_with("nvme") && !name.contains('p')).then_some(())?;
            Some((
                f.get(5)?.parse::<u64>().ok()? * 512,
                f.get(9)?.parse::<u64>().ok()? * 512,
            ))
        })
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

// ── Memory ───────────────────────────────────────────────────────────────────

/// (total_kb, available_kb)
fn read_meminfo() -> (u64, u64) {
    let s = read("/proc/meminfo").unwrap_or_default();
    let get = |k: &str| {
        s.lines()
            .find(|l| l.starts_with(k))
            .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
            .unwrap_or(0)
    };
    (get("MemTotal:"), get("MemAvailable:"))
}

// ── Processes ────────────────────────────────────────────────────────────────

struct ProcSample {
    name: String,
    ticks: u64,
    rss_bytes: u64,
    read_bytes: u64,
    write_bytes: u64,
}

fn read_processes(page_size: u64) -> HashMap<i32, ProcSample> {
    let Ok(rd) = fs::read_dir("/proc") else {
        return HashMap::new();
    };
    rd.flatten()
        .filter_map(|e| {
            let pid: i32 = e.file_name().to_str()?.parse().ok()?;
            let stat = read(e.path().join("stat"))?;
            let (head, rest) = stat.rsplit_once(')')?;
            let name = head.split_once('(')?.1.to_string();
            // `rest` starts at field 3 (state): utime=14, stime=15, rss=24.
            let f: Vec<&str> = rest.split_whitespace().collect();
            let utime: u64 = f.get(11)?.parse().ok()?;
            let stime: u64 = f.get(12)?.parse().ok()?;
            let rss: u64 = f.get(21)?.parse().ok()?;
            // /proc/<pid>/io is only readable for our own processes without root.
            let io = read(e.path().join("io")).unwrap_or_default();
            let io_get = |k: &str| {
                io.lines()
                    .find_map(|l| l.strip_prefix(k)?.trim().parse().ok())
                    .unwrap_or(0)
            };
            Some((
                pid,
                ProcSample {
                    name,
                    ticks: utime + stime,
                    rss_bytes: rss * page_size,
                    read_bytes: io_get("read_bytes:"),
                    write_bytes: io_get("write_bytes:"),
                },
            ))
        })
        .collect()
}

struct ProcState {
    name: String,
    ticks: u64,
    energy_mj: f64,
    last_seen: Instant,
    sma: TimeSma,
    last: ProcessPower,
}

// ── Sampler ──────────────────────────────────────────────────────────────────

pub struct Sampler {
    shared: Arc<Mutex<Metrics>>,
    running: Arc<AtomicBool>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Sampler {
    pub fn new(interval_ms: u64) -> Self {
        let (mem_total_kb, _) = read_meminfo();
        let initial = Metrics {
            // MemTotal excludes firmware carveouts; round up to the 8 GB the machine was sold with.
            dram_gb: ((mem_total_kb as f64 / 1048576.0 / 8.0).ceil() * 8.0) as u32,
            ssd_model: read("/sys/block/nvme0n1/device/model").unwrap_or_default(),
            ..Default::default()
        };
        let shared = Arc::new(Mutex::new(initial));
        let running = Arc::new(AtomicBool::new(true));
        let dt = Duration::from_millis(interval_ms.max(100));
        let mut handles = Vec::new();

        // Wi-Fi/Ethernet: `iw` is a subprocess, so keep it off the fast loop.
        {
            let m = shared.clone();
            let running = running.clone();
            handles.push(std::thread::spawn(move || {
                while running.load(Ordering::Relaxed) {
                    let wifi = read_wifi();
                    let eth = read_ethernet();
                    if let Ok(mut mg) = m.lock() {
                        mg.wifi_power_w = wifi.estimated_power_w;
                        mg.wifi = wifi;
                        mg.ethernet = eth;
                    }
                    std::thread::sleep(dt.max(Duration::from_secs(2)));
                }
            }));
        }

        {
            let m = shared.clone();
            let running = running.clone();
            handles.push(std::thread::spawn(move || {
                Self::run(m, running, dt);
            }));
        }

        Self {
            shared,
            running,
            handles,
        }
    }

    fn run(m: Arc<Mutex<Metrics>>, running: Arc<AtomicBool>, dt: Duration) {
        let smc = hwmon_dir("macsmc_hwmon")
            .map(|d| list_sensors(&d))
            .unwrap_or_default();
        let policies = read_policies();
        let p_pols: Vec<&Policy> = policies.iter().filter(|p| p.perf).collect();
        let e_pols: Vec<&Policy> = policies.iter().filter(|p| !p.perf).collect();
        // The TUI expects usage ordered P-cores first, then E-cores.
        let cpu_order: Vec<usize> = p_pols
            .iter()
            .chain(e_pols.iter())
            .flat_map(|p| p.cpus.iter().copied())
            .collect();
        let ecpu_clusters: Vec<CpuCluster> = e_pols
            .iter()
            .enumerate()
            .map(|(i, p)| cluster(format!("ECPU{i}"), &p.cpus, &format!("ECPU{i}_")))
            .collect();
        let p_cpus: Vec<usize> = p_pols.iter().flat_map(|p| p.cpus.iter().copied()).collect();
        let pcpu_cluster = cluster("PCPU".into(), &p_cpus, "PCPU");
        let edp = edp_connector();
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as u64;

        let mut prev_ticks = read_cpu_ticks();
        let mut prev_net = read_net_counters();
        let mut prev_disk = read_disk_counters();
        let mut prev_time = Instant::now();
        let mut procs: HashMap<i32, ProcState> = HashMap::new();
        let mut total_sma = TimeSma::new(5.0);
        let mut temp_cache: HashMap<String, f32> = HashMap::new();

        while running.load(Ordering::Relaxed) {
            std::thread::sleep(dt);
            let now = Instant::now();
            let dt_s = now.duration_since(prev_time).as_secs_f64().max(1e-3);
            prev_time = now;

            // SMC via macsmc_hwmon (labels are the raw SMC keys for temps).
            let mut temps = Vec::new();
            let mut fans = Vec::new();
            let (mut sys_w, mut soc_w, mut ac_w) = (0.0f32, 0.0f32, 0.0f32);
            let (mut ac_mv, mut ac_ma) = (0u32, 0u32);
            for s in &smc {
                let v: Option<f32> = num(&s.input);
                match s.kind.as_str() {
                    "temp" => {
                        let fresh = v.map(|v| v / 1000.0).filter(|c| *c > 0.0 && *c < 150.0);
                        if let Some(c) = fresh {
                            temp_cache.insert(s.label.clone(), c);
                        }
                        if let Some(&c) = temp_cache.get(&s.label) {
                            temps.push(TempSensor {
                                key: s.label.clone(),
                                category: temp_category(&s.label).into(),
                                value_celsius: c,
                                stale: fresh.is_none(),
                            });
                        }
                    }
                    "power" => {
                        let w = v.unwrap_or(0.0) / 1e6;
                        match s.label.as_str() {
                            "Total System Power" => sys_w = w,
                            "Heatpipe Power" => soc_w = w,
                            "AC Input Power" => ac_w = w,
                            _ => {}
                        }
                    }
                    "in" if s.label == "Charger Input Voltage" => ac_mv = v.unwrap_or(0.0) as u32,
                    "curr" if s.label == "AC Input Current" => ac_ma = v.unwrap_or(0.0) as u32,
                    "fan" => {
                        let base = s.input.to_string_lossy().replace("_input", "");
                        let actual = v.unwrap_or(0.0);
                        let min: f32 = num(format!("{base}_min")).unwrap_or(0.0);
                        let max: f32 = num(format!("{base}_max")).unwrap_or(0.0);
                        let ratio = if max > 0.0 {
                            (actual / max).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        fans.push(FanInfo {
                            id: fans.len() as u32,
                            name: s.label.clone(),
                            actual_rpm: actual,
                            min_rpm: min,
                            max_rpm: max,
                            estimated_power_w: MAX_FAN_W * ratio.powi(3),
                        });
                    }
                    _ => {}
                }
            }

            // CPU usage + frequencies.
            let cur_ticks = read_cpu_ticks();
            let cpu_usage: Vec<f32> = cpu_order
                .iter()
                .map(|c| match (prev_ticks.get(c), cur_ticks.get(c)) {
                    (Some(&p), Some(&q)) => usage_pct(p, q),
                    _ => 0.0,
                })
                .collect();
            prev_ticks = cur_ticks;

            let gpu = read_gpu_stats();
            let g = |k: &str| gpu.get(k).copied().unwrap_or(0);
            let soc = SocPower {
                ecpu_clusters: ecpu_clusters.clone(),
                pcpu_cluster: pcpu_cluster.clone(),
                gpu_util_device: g("util_device_percent") as u32,
                gpu_util_renderer: g("util_renderer_percent") as u32,
                gpu_util_tiler: g("util_tiler_percent") as u32,
                gpu_freq_mhz: (g("achieved_frequency_hz") / 1_000_000) as u32,
                ecpu_freq_mhz: avg_freq_mhz(&e_pols),
                pcpu_freq_mhz: avg_freq_mhz(&p_pols),
                total_w: soc_w,
                ..Default::default()
            };

            // Network + disk.
            let cur_net = read_net_counters();
            let (wifi_if, eth_if) = m
                .lock()
                .map(|mg| {
                    (
                        mg.wifi.interface_name.clone(),
                        mg.ethernet.interface_name.clone(),
                    )
                })
                .unwrap_or_default();
            let network = net_rate(&prev_net, &cur_net, dt_s, None);
            let wifi_network = net_rate(&prev_net, &cur_net, dt_s, Some(&wifi_if));
            let eth_network = net_rate(&prev_net, &cur_net, dt_s, Some(&eth_if));
            prev_net = cur_net;

            let cur_disk = read_disk_counters();
            let disk = DiskInfo {
                read_bytes_per_sec: cur_disk.0.saturating_sub(prev_disk.0) as f64 / dt_s,
                write_bytes_per_sec: cur_disk.1.saturating_sub(prev_disk.1) as f64 / dt_s,
            };
            prev_disk = cur_disk;
            let util = ((disk.read_bytes_per_sec + disk.write_bytes_per_sec)
                / (3_000.0 * 1024.0 * 1024.0))
                .clamp(0.0, 1.0) as f32;

            // Processes: split SoC power by each process's share of CPU time.
            let cur_procs = read_processes(page_size);
            let deltas: HashMap<i32, u64> = cur_procs
                .iter()
                .map(|(pid, s)| {
                    let d = procs
                        .get(pid)
                        .filter(|p| p.name == s.name && s.ticks >= p.ticks)
                        .map_or(0, |p| s.ticks - p.ticks);
                    (*pid, d)
                })
                .collect();
            let total_delta: u64 = deltas.values().sum();
            for (pid, s) in cur_procs {
                let d = deltas[&pid];
                let raw_w = if total_delta > 0 {
                    soc_w * d as f32 / total_delta as f32
                } else {
                    0.0
                };
                let st = procs.entry(pid).or_insert_with(|| ProcState {
                    name: s.name.clone(),
                    ticks: s.ticks,
                    energy_mj: 0.0,
                    last_seen: now,
                    sma: TimeSma::new(5.0),
                    last: ProcessPower::default(),
                });
                if st.name != s.name {
                    // PID reused by a different program.
                    st.name = s.name.clone();
                    st.energy_mj = 0.0;
                    st.sma = TimeSma::new(5.0);
                }
                st.ticks = s.ticks;
                st.last_seen = now;
                st.energy_mj += raw_w as f64 * dt_s * 1000.0;
                st.sma.push(raw_w);
                st.last = ProcessPower {
                    pid,
                    name: s.name,
                    power_w: st.sma.get(),
                    energy_mj: st.energy_mj,
                    alive: true,
                    disk_read_bytes: s.read_bytes,
                    disk_write_bytes: s.write_bytes,
                    phys_mem_bytes: s.rss_bytes,
                    net_rx_bytes: 0,
                    net_tx_bytes: 0,
                };
            }
            procs.retain(|_, p| now.duration_since(p.last_seen).as_secs() < 30);
            let mut top: Vec<ProcessPower> = procs
                .values()
                .filter(|p| p.energy_mj > 0.0)
                .map(|p| ProcessPower {
                    alive: p.last_seen == now,
                    power_w: if p.last_seen == now {
                        p.last.power_w
                    } else {
                        0.0
                    },
                    ..p.last.clone()
                })
                .collect();
            let total_energy: f64 = top.iter().map(|p| p.energy_mj).sum();
            total_sma.push(top.iter().map(|p| p.power_w).sum());
            top.sort_by(|a, b| b.energy_mj.total_cmp(&a.energy_mj));
            top.truncate(50);

            let (mem_total, mem_avail) = read_meminfo();
            let kbd = num::<f32>(format!("{KBD}/brightness"))
                .zip(num::<f32>(format!("{KBD}/max_brightness")))
                .filter(|(_, max)| *max > 0.0)
                .map_or(0.0, |(b, max)| b / max);
            let display = read_display(edp.as_deref());
            let ac_online = num::<u8>(format!("{AC}/online")) == Some(1);
            let battery = read_battery(sys_w);

            if let Ok(mut mg) = m.lock() {
                mg.soc = soc;
                mg.temperatures = temps;
                mg.fans = fans;
                mg.sys_power_w = sys_w;
                mg.adapter_power_w = ac_w;
                mg.adapter = AdapterInfo {
                    connected: ac_online,
                    watts: num::<u64>(format!("{AC}/input_power_limit"))
                        .map_or(0, |uw| (uw / 1_000_000) as u32),
                    voltage_mv: if ac_online { ac_mv } else { 0 },
                    current_ma: if ac_online { ac_ma } else { 0 },
                    is_wireless: false,
                };
                mg.battery = battery;
                mg.cpu_usage_pct = cpu_usage;
                mg.mem_used_gb = mem_total.saturating_sub(mem_avail) as f32 / 1048576.0;
                mg.network = network;
                mg.wifi_network = wifi_network;
                mg.eth_network = eth_network;
                mg.disk = disk;
                mg.ssd_power_w = SSD_IDLE_W + util * (SSD_MAX_ACTIVE_W - SSD_IDLE_W);
                mg.keyboard = KeyboardInfo {
                    brightness_pct: kbd * 100.0,
                    estimated_power_w: kbd * MAX_KEYBOARD_W,
                };
                mg.display = display;
                mg.top_processes = top;
                mg.all_procs_power_w = total_sma.get();
                mg.all_procs_energy_mj = total_energy;
            }
        }
    }

    /// Return a snapshot of the current metrics (non-blocking).
    pub fn snapshot(&self) -> Metrics {
        self.shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_pct_handles_deltas_and_resets() {
        assert_eq!(usage_pct((100, 200), (150, 300)), 50.0);
        assert_eq!(usage_pct((1000, 2000), (10, 20)), 0.0);
    }

    #[test]
    fn temp_category_maps_smc_keys() {
        assert_eq!(temp_category("Tp0C"), "CPU");
        assert_eq!(temp_category("TPMP"), "Trackpad");
        assert_eq!(temp_category("Tg08"), "GPU");
        assert_eq!(temp_category("TCHP"), "Other");
    }
}
