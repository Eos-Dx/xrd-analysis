#!/bin/bash
# Test runner for Session Container Implementation

echo "================================================================================"
echo "SESSION CONTAINER IMPLEMENTATION - TEST SUITE"
echo "================================================================================"
echo ""

# Activate conda environment
echo "Activating conda environment (eosdx)..."
eval "$(conda shell.bash hook)"
conda activate eosdx

cd /Users/sad/dev/xrd-analysis

echo ""
echo "================================================================================"
echo "LAYER 1: Session Container API Tests"
echo "================================================================================"
python -m pytest src/hardware/difra/tests/test_session_container.py -v --tb=short

echo ""
echo "================================================================================"
echo "LAYER 4: Integration Tests"
echo "================================================================================"
python -m pytest src/hardware/difra/tests/test_session_integration.py -v --tb=short

echo ""
echo "================================================================================"
echo "RUNNING ALL TESTS TOGETHER"
echo "================================================================================"
python -m pytest src/hardware/difra/tests/test_session*.py -v --tb=short

echo ""
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
python -m pytest src/hardware/difra/tests/test_session*.py --co -q | wc -l
echo "Total tests created"

echo ""
echo "✅ Test suite completed. Check results above."
