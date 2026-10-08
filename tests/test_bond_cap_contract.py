"""An optional spectral cap is validated and remains provenance-recorded."""
from pathlib import Path
import pytest
from aiida_itensor_spin.contracts import ContractError, MOMENT_CONVENTION, normalize_parameters

@pytest.mark.parametrize("cap", [0, 1, 9500])
def test_optional_cap(cap):
    parameters = normalize_parameters({"chebyshev_convention": MOMENT_CONVENTION,
        "max_bond_dimension": cap, "spectral_cutoff": 1e-8})
    assert parameters["max_bond_dimension"] == cap
    assert parameters["spectral_cutoff"] == 1e-8

@pytest.mark.parametrize("cap", [-1, True, 1.5, "9500", None])
def test_invalid_cap(cap):
    with pytest.raises(ContractError):
        normalize_parameters({"chebyshev_convention": MOMENT_CONVENTION,
            "max_bond_dimension": cap})

def test_legacy_no_cap_remains_unchanged():
    assert "max_bond_dimension" not in normalize_parameters({"chebyshev_convention": MOMENT_CONVENTION})

@pytest.mark.parametrize("driver", ["itensor-spin-segment", "itensor-spin-segment-local"])
def test_both_drivers_forward_cap_and_cutoff(driver):
    source = (Path(__file__).resolve().parents[1] / "remote" / driver).read_text()
    assert '"ITENSOR_SPIN_MAX_BOND_DIMENSION"' in source
    assert '"ITENSOR_SPIN_SPECTRAL_CUTOFF"' in source
    assert '["--spectral-cutoff", str(spectral_cutoff)]' in source
