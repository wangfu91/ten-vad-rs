use std::env;
use std::path::PathBuf;

fn main() {
    let lib_dir = "lib";
    let include_dir = "include";

    let lib_path: PathBuf;
    #[cfg(target_os = "windows")]
    {
        let native = PathBuf::from(format!("{lib_dir}/windows"));
        #[cfg(target_arch = "x86")]
        let arch = "x86";

        #[cfg(target_arch = "x86_64")]
        let arch = "x64";

        lib_path = native.join(arch);
    }

    // Link search path (for .lib)
    println!("cargo:rustc-link-search=native={}", lib_path.display());

    // Link against the DLL (via its import .lib)
    println!("cargo:rustc-link-lib=dylib=ten_vad");

    // Re-run build.rs if the header or import lib changes
    println!("cargo:rerun-if-changed={include_dir}/ten_vad.h");
    println!("cargo:rerun-if-changed={}/ten_vad.lib", lib_path.display());

    // Generate FFI bindings using bindgen
    let bindings = bindgen::Builder::default()
        .header(format!("{include_dir}/ten_vad.h"))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to $OUT_DIR/bindings.rs
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // Copy the ten_vad.dll to the target directory where the executable will be
    #[cfg(target_os = "windows")]
    {
        use std::fs;

        let dll_path = lib_path.join("ten_vad.dll");

        // Get the target directory (e.g., target/debug or target/release)
        let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
        let target_dir = env::var("CARGO_TARGET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("target"));
        let exe_dir = target_dir.join(&profile);

        // Ensure the target directory exists
        fs::create_dir_all(&exe_dir).expect("Failed to create target directory");

        let target_dll_path = exe_dir.join("ten_vad.dll");
        fs::copy(&dll_path, &target_dll_path).expect("Failed to copy ten_vad.dll");
    }
}
