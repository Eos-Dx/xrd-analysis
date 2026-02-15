///! File Converter for Bruker .gfrm Frame Files
///!
///! Converts Bruker X-ray detector .gfrm files into multiple output formats:
///! - ASCII (.dat) - Raw ADU values with baseline offset
///! - Net Intensity (.net.txt) - ADU values with baseline subtracted
///! - Photon Counts (.photons.txt) - Converted to photon counts
///! - Heatmap Image (.png) - Visual heatmap with gradient coloring
///! - Data File (.data) - Processed measurement data
///! - TIFF (.tiff) - Standard image format
///!
///! Based on the EosDx FileConverter implementation.

use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// File conversion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverterConfig {
    /// Heatmap gradient minimum value
    pub heatmap_min: i32,
    /// Heatmap gradient maximum value
    pub heatmap_max: i32,
    /// Output directory for converted files
    pub output_dir: PathBuf,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            heatmap_min: 0,
            heatmap_max: 290,
            output_dir: PathBuf::from("measurements"),
        }
    }
}

/// Results from file conversion
#[derive(Debug, Clone)]
pub struct ConversionResult {
    pub ascii_file: PathBuf,
    pub net_intensity_file: PathBuf,
    pub photon_counts_file: PathBuf,
    pub heatmap_file: PathBuf,
    pub data_file: PathBuf,
    pub tiff_file: PathBuf,
}

/// Bruker frame header parameters
#[derive(Debug, Clone)]
struct FrameHeader {
    /// Baseline offset for ADU conversion (from NEXP line)
    baseline_offset: i32,
    /// Electronic gain (e-/ADU) from CCDPARM line
    electronic_gain: f64,
    /// Optical gain (e-/photon) from CCDPARM line
    optical_gain: f64,
    /// Number of rows
    nrows: usize,
    /// Number of columns
    ncols: usize,
}

/// File converter for Bruker .gfrm files
pub struct FileConverter {
    config: ConverterConfig,
}

impl FileConverter {
    pub fn new(config: ConverterConfig) -> Self {
        Self { config }
    }

    /// Convert a .gfrm file to all output formats
    ///
    /// # Arguments
    /// * `gfrm_file` - Path to the input .gfrm file
    /// * `output_name` - Base name for output files (without extension)
    ///
    /// # Returns
    /// Paths to all generated files
    pub async fn convert_all(&self, gfrm_file: &Path, output_name: &str) -> Result<ConversionResult> {
        info!("Converting Bruker frame file: {:?}", gfrm_file);

        // Ensure output directory exists
        fs::create_dir_all(&self.config.output_dir)
            .context("Failed to create output directory")?;

        // Parse the .gfrm file
        let (header, image_data) = self.parse_gfrm_file(gfrm_file)
            .context("Failed to parse .gfrm file")?;

        // Generate output paths
        let ascii_file = self.config.output_dir.join(format!("{}.ascii", output_name));
        let net_intensity_file = self.config.output_dir.join(format!("{}_net.txt", output_name));
        let photon_counts_file = self.config.output_dir.join(format!("{}_photons.txt", output_name));
        let heatmap_file = self.config.output_dir.join(format!("{}_heatmap.png", output_name));
        let data_file = self.config.output_dir.join(format!("{}.data", output_name));
        let tiff_file = self.config.output_dir.join(format!("{}.tiff", output_name));

        // 1. Create ASCII file (raw ADU values)
        info!("Creating ASCII file...");
        self.write_ascii_file(&image_data, &ascii_file)?;

        // 2. Create Net Intensity file (baseline subtracted)
        info!("Creating Net Intensity file...");
        let net_intensity_data = self.calculate_net_intensity(&image_data, &header);
        self.write_integer_array(&net_intensity_data, &net_intensity_file)?;

        // 3. Create Photon Counts file
        info!("Creating Photon Counts file...");
        let photon_data = self.calculate_photon_counts(&image_data, &header);
        self.write_float_array(&photon_data, &photon_counts_file)?;

        // 4. Generate Heatmap Image
        info!("Generating Heatmap Image...");
        self.generate_heatmap_image(&net_intensity_data, &heatmap_file)?;

        // 5. Create Data file (metadata + processed data)
        info!("Creating Data file...");
        self.write_data_file(&header, &net_intensity_data, &photon_data, &data_file)?;

        // 6. Create TIFF file
        info!("Creating TIFF file...");
        self.write_tiff_file(&net_intensity_data, &tiff_file)?;

        info!("File conversion completed successfully");

        Ok(ConversionResult {
            ascii_file,
            net_intensity_file,
            photon_counts_file,
            heatmap_file,
            data_file,
            tiff_file,
        })
    }

    /// Parse Bruker .gfrm file
    ///
    /// This is a simplified parser. In production, you would use the actual
    /// Bruker frame format specification or integrate with Bruker's SDK.
    fn parse_gfrm_file(&self, _path: &Path) -> Result<(FrameHeader, Vec<Vec<i32>>)> {
        // TODO: Implement actual .gfrm parsing
        // For now, return dummy data
        
        debug!("Parsing .gfrm file (placeholder implementation)");
        
        // Placeholder header
        let header = FrameHeader {
            baseline_offset: 32,
            electronic_gain: 11.18,
            optical_gain: 175.0,
            nrows: 100,
            ncols: 100,
        };

        // Placeholder image data (100x100 with random values)
        let mut image_data = Vec::new();
        for _ in 0..header.nrows {
            let mut row = Vec::new();
            for _ in 0..header.ncols {
                row.push(rand::random::<i32>() % 200 + 50); // Random ADU values
            }
            image_data.push(row);
        }

        Ok((header, image_data))
    }

    /// Write ASCII file with raw ADU values
    fn write_ascii_file(&self, data: &[Vec<i32>], output_path: &Path) -> Result<()> {
        let mut content = String::new();
        
        for row in data {
            let row_str: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            content.push_str(&row_str.join(" "));
            content.push('\n');
        }

        fs::write(output_path, content)
            .context("Failed to write ASCII file")?;

        Ok(())
    }

    /// Calculate net intensity (ADU with baseline subtracted)
    ///
    /// Formula: net_intensity = raw_adu - baseline_offset
    fn calculate_net_intensity(&self, data: &[Vec<i32>], header: &FrameHeader) -> Vec<Vec<i32>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .map(|&adu| adu - header.baseline_offset)
                    .collect()
            })
            .collect()
    }

    /// Calculate photon counts from ADU values
    ///
    /// Formula: photons = net_intensity * (electronic_gain / optical_gain)
    fn calculate_photon_counts(&self, data: &[Vec<i32>], header: &FrameHeader) -> Vec<Vec<f64>> {
        let conversion_factor = header.electronic_gain / header.optical_gain;
        
        data.iter()
            .map(|row| {
                row.iter()
                    .map(|&adu| {
                        let net_intensity = adu - header.baseline_offset;
                        net_intensity as f64 * conversion_factor
                    })
                    .collect()
            })
            .collect()
    }

    /// Write integer array to file
    fn write_integer_array(&self, data: &[Vec<i32>], output_path: &Path) -> Result<()> {
        let mut content = String::new();
        
        for row in data {
            let row_str: Vec<String> = row.iter().map(|v| v.to_string()).collect();
            content.push_str(&row_str.join(" "));
            content.push('\n');
        }

        fs::write(output_path, content)
            .context("Failed to write integer array file")?;

        Ok(())
    }

    /// Write float array to file
    fn write_float_array(&self, data: &[Vec<f64>], output_path: &Path) -> Result<()> {
        let mut content = String::new();
        
        for row in data {
            let row_str: Vec<String> = row.iter().map(|v| format!("{:.4}", v)).collect();
            content.push_str(&row_str.join(" "));
            content.push('\n');
        }

        fs::write(output_path, content)
            .context("Failed to write float array file")?;

        Ok(())
    }

    /// Generate heatmap image from net intensity data
    ///
    /// Uses a 60-color gradient from black → red → yellow → white
    fn generate_heatmap_image(&self, data: &[Vec<i32>], output_path: &Path) -> Result<()> {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };

        if nrows == 0 || ncols == 0 {
            return Err(anyhow::anyhow!("Empty data array"));
        }

        // Create gradient map
        let gradient = self.create_heatmap_gradient();

        // Create image with scaled pixels (10x10 per data point)
        let pixel_size = 10;
        let img_width = (ncols * pixel_size) as u32;
        let img_height = (nrows * pixel_size) as u32;

        let mut img: RgbImage = ImageBuffer::new(img_width, img_height);

        for (row_idx, row) in data.iter().enumerate() {
            for (col_idx, &value) in row.iter().enumerate() {
                let color = self.get_heatmap_color(value, &gradient);
                
                // Fill pixel_size x pixel_size block
                for dy in 0..pixel_size {
                    for dx in 0..pixel_size {
                        let x = (col_idx * pixel_size + dx) as u32;
                        let y = (row_idx * pixel_size + dy) as u32;
                        img.put_pixel(x, y, color);
                    }
                }
            }
        }

        img.save(output_path)
            .context("Failed to save heatmap image")?;

        Ok(())
    }

    /// Create heatmap color gradient (60 colors from black to white through red/yellow)
    fn create_heatmap_gradient(&self) -> HashMap<i32, Rgb<u8>> {
        let mut gradient = HashMap::new();
        let colors = self.get_heatmap_colors();
        
        let range = self.config.heatmap_max - self.config.heatmap_min;
        let step = range / (colors.len() as i32 - 1);
        
        for (i, color) in colors.iter().enumerate() {
            let value = self.config.heatmap_min + (i as i32 * step);
            gradient.insert(value, *color);
        }

        gradient
    }

    /// Get heatmap color for a value
    fn get_heatmap_color(&self, value: i32, gradient: &HashMap<i32, Rgb<u8>>) -> Rgb<u8> {
        // Clamp value to range
        let clamped = value.clamp(self.config.heatmap_min, self.config.heatmap_max);
        
        // Find closest gradient point
        let mut best_key = self.config.heatmap_min;
        let mut best_diff = (clamped - best_key).abs();
        
        for &key in gradient.keys() {
            let diff = (clamped - key).abs();
            if diff < best_diff {
                best_diff = diff;
                best_key = key;
            }
        }
        
        *gradient.get(&best_key).unwrap_or(&Rgb([0, 0, 0]))
    }

    /// Get 60-color heatmap palette (black → red → yellow → white)
    fn get_heatmap_colors(&self) -> Vec<Rgb<u8>> {
        vec![
            Rgb([0, 0, 0]), Rgb([30, 0, 0]), Rgb([60, 0, 0]), Rgb([92, 0, 0]),
            Rgb([117, 0, 0]), Rgb([139, 0, 0]), Rgb([159, 0, 0]), Rgb([177, 0, 0]),
            Rgb([194, 0, 0]), Rgb([211, 0, 0]), Rgb([226, 0, 0]), Rgb([241, 0, 0]),
            Rgb([255, 21, 0]), Rgb([255, 40, 0]), Rgb([255, 55, 0]), Rgb([255, 67, 0]),
            Rgb([255, 78, 0]), Rgb([255, 88, 0]), Rgb([255, 97, 0]), Rgb([255, 106, 0]),
            Rgb([255, 114, 0]), Rgb([255, 122, 0]), Rgb([255, 129, 0]), Rgb([255, 137, 0]),
            Rgb([255, 144, 0]), Rgb([255, 151, 0]), Rgb([255, 157, 0]), Rgb([255, 164, 0]),
            Rgb([255, 176, 0]), Rgb([255, 182, 0]), Rgb([255, 188, 0]), Rgb([255, 193, 0]),
            Rgb([255, 199, 0]), Rgb([255, 205, 0]), Rgb([255, 210, 0]), Rgb([255, 215, 0]),
            Rgb([255, 220, 0]), Rgb([255, 226, 0]), Rgb([255, 231, 0]), Rgb([255, 236, 0]),
            Rgb([255, 241, 0]), Rgb([255, 245, 0]), Rgb([255, 250, 0]), Rgb([255, 255, 27]),
            Rgb([255, 255, 52]), Rgb([255, 255, 70]), Rgb([255, 255, 86]), Rgb([255, 255, 100]),
            Rgb([255, 255, 113]), Rgb([255, 255, 124]), Rgb([255, 255, 136]), Rgb([255, 255, 146]),
            Rgb([255, 255, 166]), Rgb([255, 255, 175]), Rgb([255, 255, 184]), Rgb([255, 255, 193]),
            Rgb([255, 255, 201]), Rgb([255, 255, 210]), Rgb([255, 255, 218]), Rgb([255, 255, 225]),
            Rgb([255, 255, 233]),
        ]
    }

    /// Write data file with metadata and processed values
    fn write_data_file(
        &self,
        header: &FrameHeader,
        net_intensity: &[Vec<i32>],
        photons: &[Vec<f64>],
        output_path: &Path,
    ) -> Result<()> {
        let mut content = String::new();
        
        // Write header metadata
        content.push_str(&format!("# Bruker Frame Data File\n"));
        content.push_str(&format!("# Baseline Offset: {}\n", header.baseline_offset));
        content.push_str(&format!("# Electronic Gain: {:.4} e-/ADU\n", header.electronic_gain));
        content.push_str(&format!("# Optical Gain: {:.4} e-/photon\n", header.optical_gain));
        content.push_str(&format!("# Conversion Factor: {:.6} photon/ADU\n", 
            header.electronic_gain / header.optical_gain));
        content.push_str(&format!("# Dimensions: {} x {}\n", header.nrows, header.ncols));
        content.push_str(&format!("# Generated: {}\n", chrono::Utc::now().to_rfc3339()));
        content.push_str("\n");

        // Write statistics
        let net_min = net_intensity.iter().flat_map(|r| r.iter()).min().unwrap_or(&0);
        let net_max = net_intensity.iter().flat_map(|r| r.iter()).max().unwrap_or(&0);
        let photon_min = photons.iter().flat_map(|r| r.iter()).min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let photon_max = photons.iter().flat_map(|r| r.iter()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);

        content.push_str(&format!("# Net Intensity Range: {} to {} ADU\n", net_min, net_max));
        content.push_str(&format!("# Photon Count Range: {:.2} to {:.2} photons\n", photon_min, photon_max));
        content.push_str("\n");

        fs::write(output_path, content)
            .context("Failed to write data file")?;

        Ok(())
    }

    /// Write TIFF file from net intensity data
    fn write_tiff_file(&self, data: &[Vec<i32>], output_path: &Path) -> Result<()> {
        let nrows = data.len();
        let ncols = if nrows > 0 { data[0].len() } else { 0 };

        if nrows == 0 || ncols == 0 {
            return Err(anyhow::anyhow!("Empty data array"));
        }

        // Normalize to 0-255 range for 8-bit grayscale
        let min_val = data.iter().flat_map(|r| r.iter()).min().unwrap_or(&0);
        let max_val = data.iter().flat_map(|r| r.iter()).max().unwrap_or(&255);
        let range = (max_val - min_val).max(1) as f64;

        let img: image::GrayImage = ImageBuffer::from_fn(
            ncols as u32,
            nrows as u32,
            |x, y| {
                let value = data[y as usize][x as usize];
                let normalized = ((value - min_val) as f64 / range * 255.0) as u8;
                image::Luma([normalized])
            },
        );

        img.save(output_path)
            .context("Failed to save TIFF file")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_file_converter_basic() {
        let config = ConverterConfig::default();
        let converter = FileConverter::new(config);

        // Test with dummy data
        let temp_dir = std::env::temp_dir().join("omniscan_test");
        std::fs::create_dir_all(&temp_dir).unwrap();

        let gfrm_file = temp_dir.join("test.gfrm");
        std::fs::write(&gfrm_file, b"dummy data").unwrap();

        let result = converter.convert_all(&gfrm_file, "test_output").await;
        assert!(result.is_ok());
    }
}
