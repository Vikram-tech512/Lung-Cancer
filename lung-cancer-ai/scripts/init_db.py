"""
scripts/init_db.py
==================
Database initialization and bootstrap seed script.
Creates tables, default administrative accounts, models registry, and sample cohort.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from backend.database.database import engine, SessionLocal, Base
from backend.database.models import User, AIModel, Dataset, Patient, Study, Lesion, Measurement
from backend.auth.security import hash_password

def init_database():
    print("[INIT] Creating PostgreSQL relational schema (20 tables)...")
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 1. Seed Admin User
        admin_email = "admin@lungcancerai.org"
        admin = db.query(User).filter(User.email == admin_email).first()
        if not admin:
            admin = User(
                email=admin_email,
                hashed_password=hash_password("AdminPass2026!"),
                full_name="Lead Clinical AI Director",
                role="ADMIN"
            )
            db.add(admin)
            print(f"[SEED] Created Admin User: {admin_email} (Pass: AdminPass2026!)")

        # 2. Seed Researcher User
        res_email = "researcher@lungcancerai.org"
        researcher = db.query(User).filter(User.email == res_email).first()
        if not researcher:
            researcher = User(
                email=res_email,
                hashed_password=hash_password("ResearcherPass2026!"),
                full_name="Senior Thoracic Radiologist",
                role="RESEARCHER"
            )
            db.add(researcher)
            print(f"[SEED] Created Researcher User: {res_email}")

        # 3. Seed AI Models in Registry
        models_data = [
            ("3D ResNet-50", "v1.0.0", "PRODUCTION", "3D ResNet-50 Bottleneck + MC Dropout (T=30)", "(1, 64, 64, 64)"),
            ("3D Swin Transformer", "v1.0.0", "PRODUCTION", "Swin3D-Tiny (shifted window attention, embed_dim=96)", "(1, 64, 64, 64)"),
            ("Hybrid CNN-ViT", "v1.0.0", "STAGING", "ResNet-50 stem + ViT-B/16 gated cross-attention fusion", "(3, 224, 224)"),
            ("3D U-Net", "v1.0.0", "PRODUCTION", "3D U-Net (4 resolution levels, channels [32, 64, 128, 256])", "(1, 128, 128, 128)"),
            ("nnU-Net v2", "v2.2.0", "STAGING", "nnU-Net 3d_fullres configuration for Dataset004_Lungs", "Isotropic 1mm³"),
            ("Radiomics XGBoost", "v1.0.0", "PRODUCTION", "XGBoost classifier (107 IBSI features + SHAP explainer)", "(107,)")
        ]
        for name, ver, stage, arch, dim in models_data:
            m = db.query(AIModel).filter(AIModel.model_name == name).first()
            if not m:
                m = AIModel(model_name=name, version=ver, stage=stage, architecture_details=arch, input_dimension=dim)
                db.add(m)
        print("[SEED] Registered 6 AI Models in Model Registry.")

        # 4. Seed Datasets
        datasets_data = [
            ("LIDC-IDRI", "v1.0", "TCIA", "10.7937/K9/TCIA.2015.11S04692", 1018),
            ("NLST", "v2.0", "NCI", "10.7937/TCIA.HMQ8-J677", 26722),
            ("NSCLC-Radiomics", "v1.0", "TCIA", "10.7937/K9/TCIA.2015.PF0M9REI", 422)
        ]
        for dname, dver, dsrc, ddoi, dsub in datasets_data:
            d = db.query(Dataset).filter(Dataset.dataset_name == dname).first()
            if not d:
                d = Dataset(dataset_name=dname, version=dver, source_url=dsrc, doi=ddoi, patient_count=dsub)
                db.add(d)
        print("[SEED] Registered 3 Clinical Cohorts in Dataset Registry.")

        # 5. Seed Demonstration Study
        demo_p = db.query(Patient).filter(Patient.research_patient_id == "LIDC-IDRI-0087").first()
        if not demo_p:
            demo_p = Patient(research_patient_id="LIDC-IDRI-0087", age=68, biological_sex="M", smoking_pack_years=35.0)
            db.add(demo_p)
            db.commit()
            db.refresh(demo_p)

        demo_s = db.query(Study).filter(Study.id == "CASE-DEMO-LIDC-0087").first()
        if not demo_s:
            demo_s = Study(
                id="CASE-DEMO-LIDC-0087",
                patient_id=demo_p.id,
                study_instance_uid="1.3.6.1.4.1.14519.5.2.1.6279.6001.1790493736364",
                study_date="2026-09-22",
                modality="CT",
                slice_count=184,
                slice_thickness=1.25,
                pixel_spacing_xy=[0.703, 0.703],
                manufacturer="GE Medical Systems",
                qc_status="PASS",
                qc_findings=["Contiguous 184 slices", "Isotropic resampling verified", "No motion rings"],
                storage_folder="studies/CASE-DEMO-LIDC-0087/original"
            )
            db.add(demo_s)
            db.commit()
            db.refresh(demo_s)

            # Seed Lesion
            l = Lesion(
                id="LESION-DEMO-01",
                study_id=demo_s.id,
                lesion_index=1,
                centroid_z=92.0,
                centroid_y=280.0,
                centroid_x=310.0,
                diameter_mm=14.8,
                volume_mm3=1696.5,
                anatomical_lobe="Right Upper Lobe (RUL)",
                preliminary_risk="High Suspicion (Lung-RADS 4B)"
            )
            db.add(l)
            db.commit()
            db.refresh(l)

            # Seed Measurement
            meas = Measurement(
                lesion_id=l.id,
                area_mm2=13125.0,
                volume_mm3=1696.5,
                effective_diameter_mm=14.8,
                sphericity=0.74,
                spiculation_index=0.42,
                compactness=0.44,
                mean_hu=38.4,
                std_hu=24.1,
                min_hu=-110.0,
                max_hu=92.0
            )
            db.add(meas)
            print("[SEED] Seeded demonstration LIDC study and lesion candidate.")

        db.commit()
        print("[SUCCESS] Database initialized successfully.")
    except Exception as e:
        db.rollback()
        print(f"[ERROR] Database initialization failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    init_database()
