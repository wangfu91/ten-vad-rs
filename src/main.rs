#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(non_upper_case_globals)]
#[allow(dead_code)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[allow(unused_imports)]
use bindings::*;

fn main() {
    println!("Hello, world!");

    // Test that the bindings are accessible
    unsafe {
        let version = ten_vad_get_version();
        if !version.is_null() {
            let version_str = std::ffi::CStr::from_ptr(version);
            println!("ten_vad version: {:?}", version_str.to_string_lossy());
        }
    }
}
