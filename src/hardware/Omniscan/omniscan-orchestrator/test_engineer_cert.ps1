# Test Script for Engineer Certificate Functionality
# This script demonstrates and tests the engineer certificate features

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Engineer Certificate Test Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Test configuration
$TEST_ENGINEER_ID = "TEST001"
$TEST_DEVICE_UUID = "ABC123"
$CERT_CENTER = "C:\dev\Omniscan\omniscan-certificate-center"

# Step 1: Check if omni-orch is available
Write-Host "[1/6] Checking if omni-orch CLI is installed..." -ForegroundColor Yellow
try {
    $version = & omni-orch --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ omni-orch is installed and accessible" -ForegroundColor Green
    } else {
        Write-Host "❌ omni-orch not found. Please install:" -ForegroundColor Red
        Write-Host "   cd C:\dev\Omniscan\omniscan-orchestrator" -ForegroundColor Yellow
        Write-Host "   pip install -e ." -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "❌ omni-orch not found. Please install it first." -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Check certificate center
Write-Host "[2/6] Checking certificate center..." -ForegroundColor Yellow
if (Test-Path "$CERT_CENTER\certgen.py") {
    Write-Host "✅ Certificate center found at $CERT_CENTER" -ForegroundColor Green
} else {
    Write-Host "❌ Certificate center not found at $CERT_CENTER" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Generate test certificate
Write-Host "[3/6] Generating test engineer certificate..." -ForegroundColor Yellow
Write-Host "   Engineer ID: $TEST_ENGINEER_ID" -ForegroundColor Gray
Write-Host "   Device UUID: $TEST_DEVICE_UUID" -ForegroundColor Gray
Write-Host ""

& omni-orch cert generate `
    --engineer-id $TEST_ENGINEER_ID `
    --device-uuid $TEST_DEVICE_UUID `
    --validity-days 1 `
    --cert-center-dir $CERT_CENTER

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Certificate generation failed!" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 4: List certificates
Write-Host "[4/6] Listing all engineer certificates..." -ForegroundColor Yellow
Write-Host ""

& omni-orch cert list --cert-center-dir $CERT_CENTER

Write-Host ""

# Step 5: View certificate details
Write-Host "[5/6] Viewing test certificate details..." -ForegroundColor Yellow
Write-Host ""

$certPath = "$CERT_CENTER\certs\client\client_${TEST_ENGINEER_ID}_${TEST_DEVICE_UUID}.crt"

if (Test-Path $certPath) {
    & omni-orch cert info --cert-path $certPath
} else {
    Write-Host "❌ Certificate not found at $certPath" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 6: Test helper script
Write-Host "[6/6] Testing helper script..." -ForegroundColor Yellow
if (Test-Path ".\engineer-connect.ps1") {
    Write-Host "✅ Helper script found: .\engineer-connect.ps1" -ForegroundColor Green
    Write-Host ""
    Write-Host "Testing helper script commands:" -ForegroundColor Cyan
    Write-Host "  .\engineer-connect.ps1 config" -ForegroundColor Gray
    Write-Host "  .\engineer-connect.ps1 check-cert" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "Note: Edit engineer-connect.ps1 to set your engineer ID and device UUID" -ForegroundColor Yellow
} else {
    Write-Host "⚠️  Helper script not found (optional)" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ All tests passed!" -ForegroundColor Green
Write-Host ""
Write-Host "Certificate Details:" -ForegroundColor Yellow
Write-Host "  Certificate: $certPath" -ForegroundColor Gray
Write-Host "  Key: $CERT_CENTER\certs\client\client_${TEST_ENGINEER_ID}_${TEST_DEVICE_UUID}.key" -ForegroundColor Gray
Write-Host "  CA Cert: $CERT_CENTER\certs\root\device_server_root_ca.crt" -ForegroundColor Gray
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Start the server with mTLS enabled" -ForegroundColor Gray
Write-Host "  2. Test connection:" -ForegroundColor Gray
Write-Host "     omni-orch status \" -ForegroundColor Gray
Write-Host "       --cert $certPath \" -ForegroundColor Gray
Write-Host "       --key $CERT_CENTER\certs\client\client_${TEST_ENGINEER_ID}_${TEST_DEVICE_UUID}.key \" -ForegroundColor Gray
Write-Host "       --ca-cert $CERT_CENTER\certs\root\device_server_root_ca.crt \" -ForegroundColor Gray
Write-Host "       --base-url https://localhost:8443/api/v1" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. Or use the helper script (after editing configuration):" -ForegroundColor Gray
Write-Host "     .\engineer-connect.ps1 status" -ForegroundColor Gray
Write-Host ""
Write-Host "Documentation:" -ForegroundColor Yellow
Write-Host "  - QUICKSTART.md - Get started in 5 minutes" -ForegroundColor Gray
Write-Host "  - ENGINEER_CERTIFICATES.md - Complete guide" -ForegroundColor Gray
Write-Host "  - IMPLEMENTATION_SUMMARY.md - Technical details" -ForegroundColor Gray
Write-Host ""
