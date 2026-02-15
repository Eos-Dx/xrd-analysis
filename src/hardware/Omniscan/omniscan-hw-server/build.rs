fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(&["proto/hub/v1/hub.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/hub/v1/hub.proto");
    Ok(())
}
