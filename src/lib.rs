#[cfg(not(any(target_os = "macos", target_os = "linux")))]
compile_error!("macpow supports macOS and Linux (Asahi/Aurora on Apple Silicon) only.");

#[cfg(target_os = "macos")]
pub mod battery;
#[cfg(target_os = "macos")]
pub mod cf_utils;
#[cfg(target_os = "macos")]
pub mod iokit_ffi;
#[cfg(target_os = "macos")]
pub mod ioreport;
#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod metrics;
#[cfg(target_os = "macos")]
pub mod peripherals;
#[cfg(target_os = "macos")]
pub mod powermetrics;
pub mod process_utils;
pub mod sma;
#[cfg(target_os = "macos")]
pub mod smc;
pub mod types;
