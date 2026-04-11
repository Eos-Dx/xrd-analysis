import pandas as pd

from xrdanalysis.data_processing.splitters import make_repeated_patient_splits, patient_splitter


def _make_patient_dataset():
    rows = []
    # 12 patients, 2 specimens each, 2 rows per specimen
    for idx in range(12):
        patient = f"patient_{idx:02d}"
        patient_positive = idx % 2 == 0
        for spec_idx in range(2):
            specimen = f"specimen_{idx:02d}_{spec_idx}"
            specimen_positive = patient_positive if spec_idx == 0 else False
            for rep in range(2):
                rows.append(
                    {
                        "patientId": patient,
                        "specimenId": specimen,
                        "target_cancer_bn": bool(specimen_positive),
                        "feature": float(idx + spec_idx + rep / 10.0),
                    }
                )
    return pd.DataFrame(rows)


def test_patient_splitter_group_exclusivity_and_patient_level_stratification():
    df = _make_patient_dataset()
    y = df["target_cancer_bn"]
    X_tr, X_te, y_tr, y_te = patient_splitter(
        df,
        y,
        patient_col="patientId",
        test_size=0.3,
        random_state=32,
    )

    train_patients = set(X_tr["patientId"].astype(str).unique())
    test_patients = set(X_te["patientId"].astype(str).unique())
    assert train_patients.isdisjoint(test_patients)
    assert len(X_tr) > 0 and len(X_te) > 0

    train_patient_labels = X_tr.groupby("patientId")["target_cancer_bn"].any().astype(int)
    test_patient_labels = X_te.groupby("patientId")["target_cancer_bn"].any().astype(int)
    assert train_patient_labels.nunique() == 2
    assert test_patient_labels.nunique() == 2


def test_make_repeated_patient_splits_is_deterministic_and_group_exclusive():
    df = _make_patient_dataset()
    splits1, patient_df1 = make_repeated_patient_splits(
        df, patient_col="patientId", label_col="target_cancer_bn", n_splits=5, test_size=0.3, random_state=7
    )
    splits2, patient_df2 = make_repeated_patient_splits(
        df, patient_col="patientId", label_col="target_cancer_bn", n_splits=5, test_size=0.3, random_state=7
    )

    assert patient_df1.equals(patient_df2)
    assert splits1 == splits2
    assert len(splits1) == 5

    for split in splits1:
        train_patients = set(split["train_patients"])
        test_patients = set(split["test_patients"])
        assert train_patients.isdisjoint(test_patients)
