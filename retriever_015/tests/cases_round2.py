from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metadata import Metadata

# ------------------------------------------------------------
# test_case_15
# ------------------------------------------------------------
test_case_15 = {
    "name": 'case_15_treatment_effectiveness',
    "query": 'Does metformin improve outcomes in patients with type 2 diabetes?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Metformin and Clinical Outcomes in Type 2 Diabetes',
            year='2024',
            source_type='clinical_trial',
            id='paper_1501',
            text_summary='A clinical trial evaluating the effects of metformin on glycemic control and clinical outcomes in patients with type 2 diabetes.',
            text='We evaluated whether metformin treatment improves clinical outcomes in adults with type 2 diabetes.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Treatment Options for Type 2 Diabetes',
            year='2023',
            source_type='review',
            id='paper_1502',
            text_summary='A review of pharmacological treatments for type 2 diabetes.',
            text='Several glucose-lowering therapies are available for patients with type 2 diabetes.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Metformin: Mechanisms of Action',
            year='2022',
            source_type='paper',
            id='paper_1503',
            text_summary='Mechanisms through which metformin affects glucose metabolism.',
            text='Metformin reduces hepatic glucose production and improves insulin sensitivity.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Diabetes Mellitus: An Overview',
            year='2020',
            source_type='review',
            id='paper_1504',
            text_summary='General overview of diabetes mellitus.',
            text='Diabetes is a chronic metabolic disease affecting millions of people.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3, 'first_id': 'paper_1501', 'order_before': [('paper_1501', 'paper_1502')]}
,
}

# ------------------------------------------------------------
# test_case_16
# ------------------------------------------------------------
test_case_16 = {
    "name": 'case_16_drug_adverse_effects',
    "query": 'What are the adverse effects of statins?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Adverse Effects of Statin Therapy',
            year='2024',
            source_type='systematic_review',
            id='paper_1601',
            text_summary='Systematic review of adverse effects associated with statin therapy.',
            text='The review evaluates muscle symptoms, liver abnormalities, and other adverse effects of statins.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Statins and Cardiovascular Risk Reduction',
            year='2023',
            source_type='meta_analysis',
            id='paper_1602',
            text_summary='Meta-analysis of cardiovascular benefits associated with statin therapy.',
            text='Statins significantly reduce cardiovascular events in high-risk patients.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Statin Mechanisms',
            year='2022',
            source_type='paper',
            id='paper_1603',
            text_summary='Mechanisms of HMG-CoA reductase inhibition by statins.',
            text='Statins inhibit cholesterol synthesis by blocking HMG-CoA reductase.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Statins',
            year='2024',
            source_type='news',
            id='paper_1604',
            text_summary='News article discussing the use of statins.',
            text='Statins are widely prescribed medications for cardiovascular disease prevention.',
            image_summary=None,
            image_base_64=None,
            evidence_level='low',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_1601',
     'order_before': [('paper_1601', 'paper_1602'), ('paper_1602', 'paper_1603')]}
,
}

# ------------------------------------------------------------
# test_case_17
# ------------------------------------------------------------
test_case_17 = {
    "name": 'case_17_risk_factor',
    "query": 'Does obesity increase the risk of breast cancer?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Obesity and Breast Cancer Risk',
            year='2024',
            source_type='meta_analysis',
            id='paper_1701',
            text_summary='Meta-analysis examining the association between obesity and breast cancer risk.',
            text='Higher body mass index is associated with increased risk of breast cancer in several populations.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Obesity and Cancer',
            year='2023',
            source_type='review',
            id='paper_1702',
            text_summary='Review of associations between obesity and multiple cancers.',
            text='Obesity is associated with several cancer types.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Breast Cancer Risk Factors',
            year='2022',
            source_type='review',
            id='paper_1703',
            text_summary='Overview of established and suspected breast cancer risk factors.',
            text='Age, family history, hormonal factors, reproductive history, and obesity may influence breast cancer risk.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Obesity and Cardiovascular Disease',
            year='2024',
            source_type='meta_analysis',
            id='paper_1704',
            text_summary='Meta-analysis of obesity and cardiovascular disease.',
            text='Obesity is associated with increased cardiovascular disease risk.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_1701',
     'excludes_ids': ['paper_1704'],
     'order_before': [('paper_1701', 'paper_1702')]}
,
}

# ------------------------------------------------------------
# test_case_18
# ------------------------------------------------------------
test_case_18 = {
    "name": 'case_18_negative_evidence',
    "query": 'Does vitamin D supplementation reduce the risk of cardiovascular disease?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Vitamin D Supplementation and Cardiovascular Disease',
            year='2024',
            source_type='clinical_trial',
            id='paper_1801',
            text_summary='Randomized trial evaluating whether vitamin D supplementation reduces cardiovascular disease risk.',
            text='Vitamin D supplementation did not significantly reduce the incidence of major cardiovascular events.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Vitamin D and Cardiovascular Health',
            year='2022',
            source_type='review',
            id='paper_1802',
            text_summary='Review of associations between vitamin D status and cardiovascular health.',
            text='Observational studies have reported associations between vitamin D levels and cardiovascular outcomes.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Vitamin D and Bone Health',
            year='2024',
            source_type='clinical_trial',
            id='paper_1803',
            text_summary='Clinical trial evaluating vitamin D supplementation for bone health.',
            text='Vitamin D supplementation improves bone mineral density in selected populations.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3, 'first_id': 'paper_1801', 'order_before': [('paper_1801', 'paper_1802')]}
,
}

# ------------------------------------------------------------
# test_case_19
# ------------------------------------------------------------
test_case_19 = {
    "name": 'case_19_medical_abbreviation',
    "query": 'What are the risk factors for myocardial infarction?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Risk Factors for Myocardial Infarction',
            year='2024',
            source_type='review',
            id='paper_1901',
            text_summary='Review of major risk factors for myocardial infarction.',
            text='Hypertension, smoking, diabetes, and dyslipidemia are major risk factors for myocardial infarction.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Risk Factors for Acute Myocardial Infarction (AMI)',
            year='2023',
            source_type='paper',
            id='paper_1902',
            text_summary='Analysis of risk factors associated with acute myocardial infarction.',
            text='This study evaluates risk factors for AMI in adults.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Risk Factors for Stroke',
            year='2024',
            source_type='review',
            id='paper_1903',
            text_summary='Review of stroke risk factors.',
            text='Hypertension and smoking are important risk factors for stroke.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Cardiovascular Disease Risk',
            year='2022',
            source_type='review',
            id='paper_1904',
            text_summary='General review of cardiovascular disease risk factors.',
            text='Many cardiovascular diseases share common risk factors.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_1901',
     'contains_ids': ['paper_1902'],
     'excludes_ids': ['paper_1903']}
,
}

# ------------------------------------------------------------
# test_case_20
# ------------------------------------------------------------
test_case_20 = {
    "name": 'case_20_medical_synonym',
    "query": 'How effective is treatment for hypertension?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Antihypertensive Therapy and Blood Pressure Control',
            year='2024',
            source_type='clinical_trial',
            id='paper_2001',
            text_summary='Clinical trial evaluating antihypertensive therapy for blood pressure control.',
            text='Antihypertensive treatment significantly reduced systolic blood pressure.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Hypertension Treatment Guidelines',
            year='2023',
            source_type='guideline',
            id='paper_2002',
            text_summary='Clinical guidelines for the management of hypertension.',
            text='Lifestyle modification and antihypertensive medications are recommended for blood pressure control.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Hypertension Epidemiology',
            year='2024',
            source_type='paper',
            id='paper_2003',
            text_summary='Epidemiology of hypertension worldwide.',
            text='Hypertension affects a large proportion of the adult population.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Blood Pressure Measurement Techniques',
            year='2022',
            source_type='paper',
            id='paper_2004',
            text_summary='Methods for measuring blood pressure.',
            text='Accurate blood pressure measurement is important in clinical practice.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_2001',
     'contains_ids': ['paper_2002'],
     'excludes_ids': ['paper_2004']}
,
}

# ------------------------------------------------------------
# test_case_21
# ------------------------------------------------------------
test_case_21 = {
    "name": 'case_21_population_specific',
    "query": 'Does aspirin prevent cardiovascular events in elderly patients?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Aspirin for Primary Prevention in Older Adults',
            year='2024',
            source_type='clinical_trial',
            id='paper_2101',
            text_summary='Clinical trial evaluating aspirin for primary prevention of cardiovascular events in older adults.',
            text='Aspirin did not significantly reduce cardiovascular events among adults aged 70 years or older.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Aspirin for Cardiovascular Prevention',
            year='2023',
            source_type='meta_analysis',
            id='paper_2102',
            text_summary='Meta-analysis of aspirin for cardiovascular prevention in adults.',
            text='Low-dose aspirin may reduce cardiovascular events in selected populations.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Aspirin in Young Adults',
            year='2022',
            source_type='clinical_trial',
            id='paper_2103',
            text_summary='Trial evaluating aspirin use in younger adults.',
            text='Aspirin was evaluated for prevention of cardiovascular events in adults younger than 50 years.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Aspirin Pharmacology',
            year='2021',
            source_type='paper',
            id='paper_2104',
            text_summary='Pharmacological mechanisms of aspirin.',
            text='Aspirin inhibits platelet aggregation through irreversible COX-1 inhibition.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3, 'first_id': 'paper_2101', 'order_before': [('paper_2101', 'paper_2103')]}
,
}

# ------------------------------------------------------------
# test_case_22
# ------------------------------------------------------------
test_case_22 = {
    "name": 'case_22_disease_subtype',
    "query": 'What treatments are effective for metastatic breast cancer?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Treatment of Metastatic Breast Cancer',
            year='2024',
            source_type='systematic_review',
            id='paper_2201',
            text_summary='Systematic review of systemic treatments for metastatic breast cancer.',
            text='Endocrine therapy, chemotherapy, targeted therapy, and immunotherapy are used to treat metastatic breast cancer.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Treatment of Early-Stage Breast Cancer',
            year='2023',
            source_type='systematic_review',
            id='paper_2202',
            text_summary='Review of treatment options for early-stage breast cancer.',
            text='Surgery, radiation, and systemic therapy are used in early-stage breast cancer.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Breast Cancer Treatment',
            year='2024',
            source_type='review',
            id='paper_2203',
            text_summary='General overview of breast cancer treatment.',
            text='Treatment depends on tumor stage, receptor status, and patient characteristics.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Metastatic Lung Cancer',
            year='2024',
            source_type='systematic_review',
            id='paper_2204',
            text_summary='Systematic review of metastatic lung cancer treatment.',
            text='Systemic therapy is used for metastatic lung cancer.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_2201',
     'excludes_ids': ['paper_2204'],
     'order_before': [('paper_2201', 'paper_2202')]}
,
}

# ------------------------------------------------------------
# test_case_23
# ------------------------------------------------------------
test_case_23 = {
    "name": 'case_23_biomarker',
    "query": 'Is troponin useful for diagnosing myocardial infarction?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Cardiac Troponin for Diagnosis of Myocardial Infarction',
            year='2024',
            source_type='systematic_review',
            id='paper_2301',
            text_summary='Systematic review evaluating cardiac troponin for diagnosing myocardial infarction.',
            text='High-sensitivity cardiac troponin improves the diagnosis of myocardial infarction.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Biomarkers in Acute Coronary Syndrome',
            year='2023',
            source_type='review',
            id='paper_2302',
            text_summary='Review of biomarkers used in acute coronary syndrome.',
            text='Troponin, CK-MB, and other biomarkers can support diagnosis.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Troponin in Chronic Kidney Disease',
            year='2022',
            source_type='paper',
            id='paper_2303',
            text_summary='Study of elevated troponin levels in chronic kidney disease.',
            text='Troponin concentrations may be elevated in patients with chronic kidney disease.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Diagnosis of Myocardial Infarction',
            year='2024',
            source_type='guideline',
            id='paper_2304',
            text_summary='Clinical diagnostic criteria for myocardial infarction.',
            text='Diagnosis incorporates symptoms, ECG findings, biomarkers, and imaging.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3, 'first_id': 'paper_2301', 'order_before': [('paper_2301', 'paper_2303')]}
,
}

# ------------------------------------------------------------
# test_case_24
# ------------------------------------------------------------
test_case_24 = {
    "name": 'case_24_medical_image_summary',
    "query": 'What does a chest CT showing pulmonary embolism look like?',
    "k1": 3,
    "metadatas": [
        Metadata(
            title='Computed Tomography Findings in Pulmonary Embolism',
            year='2024',
            source_type='paper',
            id='paper_2401',
            text_summary='Imaging findings of pulmonary embolism on chest CT.',
            text='Chest CT pulmonary angiography demonstrates filling defects in pulmonary arteries.',
            image_summary=['Chest CT image showing a pulmonary arterial filling defect.', 'Axial CT image demonstrating pulmonary embolism.'],
            image_base_64=['img_ct_pe_1', 'img_ct_pe_2'],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Pulmonary Embolism Diagnosis',
            year='2023',
            source_type='review',
            id='paper_2402',
            text_summary='Review of diagnostic methods for pulmonary embolism.',
            text='CT pulmonary angiography is commonly used to diagnose pulmonary embolism.',
            image_summary=['Diagnostic algorithm for suspected pulmonary embolism.'],
            image_base_64=['img_algorithm'],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Chest CT in Pneumonia',
            year='2024',
            source_type='paper',
            id='paper_2403',
            text_summary='CT findings of pneumonia.',
            text='Chest CT can demonstrate ground-glass opacities and consolidation in pneumonia.',
            image_summary=['Chest CT showing bilateral ground-glass opacities.'],
            image_base_64=['img_pneumonia'],
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Pulmonary Embolism Treatment',
            year='2022',
            source_type='review',
            id='paper_2404',
            text_summary='Treatment options for pulmonary embolism.',
            text='Anticoagulation is the mainstay of treatment for pulmonary embolism.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 3,
     'first_id': 'paper_2401',
     'order_before': [('paper_2401', 'paper_2402')],
     'excludes_ids': ['paper_2403']}
,
}

# ------------------------------------------------------------
# test_case_25
# ------------------------------------------------------------
test_case_25 = {
    "name": 'case_25_multi_factor_medical_query',
    "query": 'What is the association between hypertension, diabetes, and stroke risk in older adults?',
    "k1": 4,
    "metadatas": [
        Metadata(
            title='Hypertension and Diabetes as Risk Factors for Stroke in Older Adults',
            year='2024',
            source_type='cohort_study',
            id='paper_2501',
            text_summary='Cohort study examining hypertension and diabetes as predictors of stroke among older adults.',
            text='Both hypertension and diabetes were independently associated with increased stroke risk in older adults.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Hypertension and Stroke Risk',
            year='2023',
            source_type='meta_analysis',
            id='paper_2502',
            text_summary='Meta-analysis of hypertension and stroke risk.',
            text='Hypertension is a major risk factor for stroke.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Diabetes and Stroke',
            year='2022',
            source_type='review',
            id='paper_2503',
            text_summary='Review of diabetes and stroke risk.',
            text='Diabetes is associated with increased risk of ischemic stroke.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Stroke in Young Adults',
            year='2024',
            source_type='paper',
            id='paper_2504',
            text_summary='Risk factors for stroke among younger adults.',
            text='Risk factors for stroke differ between younger and older populations.',
            image_summary=None,
            image_base_64=None,
            evidence_level='medium',
            last_retrieved_at='2026-08-01',
        ),
        Metadata(
            title='Hypertension and Diabetes Management',
            year='2024',
            source_type='guideline',
            id='paper_2505',
            text_summary='Clinical management of hypertension and diabetes.',
            text='Blood pressure and glucose control are important in reducing cardiovascular risk.',
            image_summary=None,
            image_base_64=None,
            evidence_level='high',
            last_retrieved_at='2026-08-01',
        ),
    ],
    "expect": {'len': 4,
     'first_id': 'paper_2501',
     'contains_ids': ['paper_2502', 'paper_2503'],
     'excludes_ids': ['paper_2504']}
,
}

ROUND2_CASES = [
    test_case_15,
    test_case_16,
    test_case_17,
    test_case_18,
    test_case_19,
    test_case_20,
    test_case_21,
    test_case_22,
    test_case_23,
    test_case_24,
    test_case_25,
]
