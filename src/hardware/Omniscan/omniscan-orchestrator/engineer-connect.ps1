# Engineer Certificate Authentication Helper Script
# Simplifies omni-orch commands by auto-managing certificates

param(
    [Parameter(Position=0, Mandatory=$false)]
    [string]$Command = "help",
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$RemainingArgs
)

# Configuration - EDIT THESE VALUES
$ENGINEER_ID = "ENG001"  # Your engineer ID
$DEVICE_UUID = "ABC123"  # Target device UUID
$CERT_DIR = "C:\dev\Omniscan\omniscan-certificate-center\certs"
$BASE_URL = "https://localhost:8443/api/v1"

# Certificate paths
$cert = "$CERT_DIR\client\client_${ENGINEER_ID}_${DEVICE_UUID}.crt"
$key = "$CERT_DIR\client\client_${ENGINEER_ID}_${DEVICE_UUID}.key"
$ca = "$CERT_DIR\root\device_server_root_ca.crt"

# Helper function to display usage
function Show-Help {
    Write-Host ""
    Write-Host "Engineer Certificate Authentication Helper" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Configuration:" -ForegroundColor Yellow
    Write-Host "  Engineer ID: $ENGINEER_ID"
    Write-Host "  Device UUID: $DEVICE_UUID"
    Write-Host "  Certificate: $cert"
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor Yellow
    Write-Host "  .\engineer-connect.ps1 <command> [options]"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\engineer-connect.ps1 status"
    Write-Host "  .\engineer-connect.ps1 enter-maintenance --ttl 900"
    Write-Host "  .\engineer-connect.ps1 get-config"
    Write-Host "  .\engineer-connect.ps1 exit-maintenance"
    Write-Host "  .\engineer-connect.ps1 cert list"
    Write-Host ""
    Write-Host "Special Commands:" -ForegroundColor Yellow
    Write-Host "  help       - Show this help message"
    Write-Host "  config     - Show current configuration"
    Write-Host "  check-cert - Check certificate status"
    Write-Host "  regen-cert - Regenerate certificate"
    Write-Host ""
}

# Helper function to show configuration
function Show-Config {
    Write-Host ""
    Write-Host "Current Configuration:" -ForegroundColor Cyan
    Write-Host "=====================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Engineer ID  : $ENGINEER_ID"
    Write-Host "Device UUID  : $DEVICE_UUID"
    Write-Host "Base URL     : $BASE_URL"
    Write-Host "Cert Directory: $CERT_DIR"
    Write-Host ""
    Write-Host "Certificate  : $cert"
    Write-Host "Private Key  : $key"
    Write-Host "CA Certificate: $ca"
    Write-Host ""
    
    if (Test-Path $cert) {
        $certInfo = Get-ChildItem $cert
        Write-Host "Certificate Status:" -ForegroundColor Green
        Write-Host "  Exists: Yes"
        Write-Host "  Created: $($certInfo.CreationTime)"
        Write-Host "  Modified: $($certInfo.LastWriteTime)"
        $age = (Get-Date) - $certInfo.LastWriteTime
        Write-Host "  Age: $([Math]::Round($age.TotalHours, 1)) hours"
        
        if ($age.TotalDays -gt 1) {
            Write-Host "  WARNING: Certificate is older than 1 day!" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Certificate Status:" -ForegroundColor Red
        Write-Host "  Exists: No"
        Write-Host "  Run '.\engineer-connect.ps1 regen-cert' to generate"
    }
    Write-Host ""
}

# Helper function to check certificate
function Check-Certificate {
    Write-Host ""
    Write-Host "Checking Certificate..." -ForegroundColor Cyan
    Write-Host ""
    
    if (-not (Test-Path $cert)) {
        Write-Host "❌ Certificate not found: $cert" -ForegroundColor Red
        Write-Host ""
        Write-Host "Generate a certificate with:" -ForegroundColor Yellow
        Write-Host "  .\engineer-connect.ps1 regen-cert"
        Write-Host ""
        return $false
    }
    
    Write-Host "✅ Certificate exists" -ForegroundColor Green
    
    # Check age
    $certInfo = Get-ChildItem $cert
    $age = (Get-Date) - $certInfo.LastWriteTime
    
    if ($age.TotalDays -gt 1) {
        Write-Host "⚠️  Certificate is $([Math]::Round($age.TotalDays, 1)) days old" -ForegroundColor Yellow
        Write-Host "   Consider regenerating with: .\engineer-connect.ps1 regen-cert" -ForegroundColor Yellow
    } else {
        Write-Host "✅ Certificate age: $([Math]::Round($age.TotalHours, 1)) hours" -ForegroundColor Green
    }
    
    # Try to get more details using omni-orch
    Write-Host ""
    Write-Host "Certificate Details:" -ForegroundColor Cyan
    omni-orch cert info --cert-path $cert 2>$null
    
    Write-Host ""
    return $true
}

# Helper function to regenerate certificate
function Regenerate-Certificate {
    Write-Host ""
    Write-Host "Generating Engineer Certificate..." -ForegroundColor Cyan
    Write-Host ""
    
    omni-orch cert generate --engineer-id $ENGINEER_ID --device-uuid $DEVICE_UUID --validity-days 1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Certificate generated successfully!" -ForegroundColor Green
        Write-Host ""
    } else {
        Write-Host ""
        Write-Host "❌ Certificate generation failed!" -ForegroundColor Red
        Write-Host ""
    }
}

# Main script logic
switch ($Command.ToLower()) {
    "help" {
        Show-Help
        exit 0
    }
    "config" {
        Show-Config
        exit 0
    }
    "check-cert" {
        Check-Certificate
        exit 0
    }
    "regen-cert" {
        Regenerate-Certificate
        exit 0
    }
    default {
        # Check if certificate exists, offer to generate if not
        if (-not (Test-Path $cert)) {
            Write-Host ""
            Write-Host "⚠️  Certificate not found: $cert" -ForegroundColor Yellow
            Write-Host ""
            $response = Read-Host "Generate certificate now? (Y/n)"
            
            if ($response -eq "" -or $response -match "^[Yy]") {
                Regenerate-Certificate
            } else {
                Write-Host "Cannot proceed without certificate. Exiting." -ForegroundColor Red
                exit 1
            }
        }
        
        # Check certificate age
        $certInfo = Get-ChildItem $cert -ErrorAction SilentlyContinue
        if ($certInfo) {
            $age = (Get-Date) - $certInfo.LastWriteTime
            if ($age.TotalDays -gt 1) {
                Write-Host "⚠️  Certificate is $([Math]::Round($age.TotalDays, 1)) days old" -ForegroundColor Yellow
                Write-Host "   Consider regenerating: .\engineer-connect.ps1 regen-cert" -ForegroundColor Yellow
                Write-Host ""
            }
        }
        
        # Build and execute the omni-orch command
        $allArgs = @($Command) + $RemainingArgs + @(
            "--cert", $cert,
            "--key", $key,
            "--ca-cert", $ca,
            "--base-url", $BASE_URL
        )
        
        Write-Host "Executing: omni-orch $($allArgs -join ' ')" -ForegroundColor DarkGray
        Write-Host ""
        
        & omni-orch $allArgs
        exit $LASTEXITCODE
    }
}
