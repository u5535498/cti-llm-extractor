"""
Intra-annotator reliability metrics.
Compares original vs re-annotate.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from krippendorff import alpha

def load_json(path: Path):
    """Load JSON file."""
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def extract_atomic_claims(doc):
    """Flatten to atomic claims for comparison (robust to missing fields)."""
    claims = []
    
    # IoCs (all types, safe access)
    iocs = []
    iocs.extend(doc.get('indicators_of_compromise', {}).get('ip_addresses', []))
    iocs.extend(doc.get('indicators_of_compromise', {}).get('domains', []))
    iocs.extend(doc.get('indicators_of_compromise', {}).get('file_hashes', []))
    iocs.extend(doc.get('indicators_of_compromise', {}).get('urls', []))
    claims.extend([f"IOC:{ioc.get('value', '')}" for ioc in iocs if ioc.get('value')])
    
    # TTPs (safe access)
    ttps = doc.get('ttps', [])
    claims.extend([f"TTP:{ttp.get('mitre_id', ttp.get('tactic', 'unknown'))}" for ttp in ttps])
    
    # Timeline (safe access)
    timeline = doc.get('timeline', [])
    claims.extend([f"Timeline:{evt.get('date', '')}" for evt in timeline if evt.get('date')])
    
    return claims

def compute_agreement_metrics(orig_path: Path, reann_path: Path):
    """Compute Krippendorff α + field-level F1."""
    orig = load_json(orig_path)
    reann = load_json(reann_path)
    
    orig_claims = extract_atomic_claims(orig)
    reann_claims = extract_atomic_claims(reann)
    
    # Krippendorff's alpha (nominal)
    reliability_data = [[c for c in orig_claims], [c for c in reann_claims]]
    kripp_alpha = alpha(reliability_data, level_of_measurement='nominal')
    
    # F1 (micro-averaged)
    all_labels = set(orig_claims + reann_claims)
    orig_vec = [1 if c in orig_claims else 0 for c in all_labels]
    reann_vec = [1 if c in reann_claims else 0 for c in all_labels]
    f1 = precision_recall_fscore_support(orig_vec, reann_vec, average='micro', zero_division=0)[2]
    
    # Simple stats
    orig_count = len(orig_claims)
    reann_count = len(reann_claims)
    
    return {
        'doc_id': orig_path.stem,
        'orig_claims': orig_count,
        'reann_claims': reann_count,
        'kripp_alpha': kripp_alpha,
        'f1_micro': f1,
        'agreement_pct': len(set(orig_claims) & set(reann_claims)) / len(set(orig_claims) | set(reann_claims))
    }

if __name__ == "__main__":
    docs = ["DOC002", "DOC006", "DOC010"]
    results = []
    
    for doc in docs:
        orig = f"data/schema_testing/{doc}_pilot.json"
        reann = f"data/schema_testing/{doc}_reannotate.json"
        
        metrics = compute_agreement_metrics(Path(orig), Path(reann))
        results.append(metrics)
        print(f"{doc}: α={metrics['kripp_alpha']:.3f}, F1={metrics['f1_micro']:.3f}")
    
    # Summary
    mean_alpha = np.mean([r['kripp_alpha'] for r in results])
    print(f"\nMean Krippendorff α: {mean_alpha:.3f} (≥0.67 = acceptable)")
