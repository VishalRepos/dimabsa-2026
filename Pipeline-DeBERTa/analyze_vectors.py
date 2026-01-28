"""
Vector Analysis Utilities
Analyze and visualize extracted vectors
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


def load_vectors(jsonl_file):
    """Load vectors from JSONL file"""
    data = {'ids': [], 'texts': [], 'vectors': [], 'metadata': []}
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            data['ids'].append(item['id'])
            data['texts'].append(item.get('text', ''))
            data['vectors'].append(item['vector'])
            data['metadata'].append({k: v for k, v in item.items() 
                                    if k not in ['id', 'text', 'vector']})
    
    data['vectors'] = np.array(data['vectors'])
    return data


def visualize_vectors_pca(vectors, labels=None, title='Vector Embeddings (PCA)'):
    """Visualize vectors using PCA"""
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    plt.figure(figsize=(10, 6))
    if labels is not None:
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                            c=labels, cmap='RdYlGn', alpha=0.6)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def visualize_vectors_tsne(vectors, labels=None, title='Vector Embeddings (t-SNE)'):
    """Visualize vectors using t-SNE"""
    tsne = TSNE(n_components=2, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(10, 6))
    if labels is not None:
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                            c=labels, cmap='RdYlGn', alpha=0.6)
        plt.colorbar(scatter, label='Label')
    else:
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def find_similar_samples(query_vector, all_vectors, all_texts, top_k=5):
    """Find most similar samples to query"""
    similarities = cosine_similarity([query_vector], all_vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': int(idx),
            'text': all_texts[idx],
            'similarity': float(similarities[idx])
        })
    return results


def analyze_va_distribution(metadata):
    """Analyze valence-arousal distribution"""
    valences = [m.get('valence', None) for m in metadata if 'valence' in m]
    arousals = [m.get('arousal', None) for m in metadata if 'arousal' in m]
    
    if not valences:
        print("No VA scores found in metadata")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Valence distribution
    axes[0].hist(valences, bins=20, color='blue', alpha=0.7)
    axes[0].set_xlabel('Valence')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Valence Distribution')
    axes[0].axvline(np.mean(valences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(valences):.2f}')
    axes[0].legend()
    
    # Arousal distribution
    axes[1].hist(arousals, bins=20, color='green', alpha=0.7)
    axes[1].set_xlabel('Arousal')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Arousal Distribution')
    axes[1].axvline(np.mean(arousals), color='red', linestyle='--',
                    label=f'Mean: {np.mean(arousals):.2f}')
    axes[1].legend()
    
    # VA scatter
    axes[2].scatter(valences, arousals, alpha=0.5)
    axes[2].set_xlabel('Valence')
    axes[2].set_ylabel('Arousal')
    axes[2].set_title('Valence-Arousal Space')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt


def compute_statistics(vectors, metadata):
    """Compute vector statistics"""
    stats = {
        'num_samples': len(vectors),
        'vector_dim': vectors.shape[1],
        'mean_norm': float(np.mean(np.linalg.norm(vectors, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(vectors, axis=1))),
        'mean_vector': vectors.mean(axis=0).tolist(),
        'std_vector': vectors.std(axis=0).tolist(),
    }
    
    # VA statistics if available
    valences = [m.get('valence') for m in metadata if 'valence' in m]
    arousals = [m.get('arousal') for m in metadata if 'arousal' in m]
    
    if valences:
        stats['valence_mean'] = float(np.mean(valences))
        stats['valence_std'] = float(np.std(valences))
        stats['arousal_mean'] = float(np.mean(arousals))
        stats['arousal_std'] = float(np.std(arousals))
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors_file', required=True, help='JSONL file with vectors')
    parser.add_argument('--output_dir', default='analysis', help='Output directory for plots')
    parser.add_argument('--max_samples', type=int, default=1000, help='Max samples for visualization')
    args = parser.parse_args()
    
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load vectors
    print(f"Loading vectors from {args.vectors_file}...")
    data = load_vectors(args.vectors_file)
    
    print(f"Loaded {len(data['ids'])} samples")
    print(f"Vector dimension: {data['vectors'].shape[1]}")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(data['vectors'], data['metadata'])
    print(f"Mean vector norm: {stats['mean_norm']:.4f} ± {stats['std_norm']:.4f}")
    
    if 'valence_mean' in stats:
        print(f"Mean valence: {stats['valence_mean']:.2f} ± {stats['valence_std']:.2f}")
        print(f"Mean arousal: {stats['arousal_mean']:.2f} ± {stats['arousal_std']:.2f}")
    
    # Save statistics
    with open(f'{args.output_dir}/statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics saved to {args.output_dir}/statistics.json")
    
    # Visualizations (use subset if too many samples)
    n_samples = min(args.max_samples, len(data['vectors']))
    vectors_subset = data['vectors'][:n_samples]
    
    # PCA visualization
    print(f"\nGenerating PCA visualization ({n_samples} samples)...")
    valences = [m.get('valence', 5.0) for m in data['metadata'][:n_samples]]
    plt_pca = visualize_vectors_pca(vectors_subset, valences, 
                                    'Vector Embeddings (PCA) - Colored by Valence')
    plt_pca.savefig(f'{args.output_dir}/vectors_pca.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {args.output_dir}/vectors_pca.png")
    
    # VA distribution
    if 'valence' in data['metadata'][0]:
        print("\nGenerating VA distribution plots...")
        plt_va = analyze_va_distribution(data['metadata'])
        plt_va.savefig(f'{args.output_dir}/va_distribution.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {args.output_dir}/va_distribution.png")
    
    print("\n✓ Analysis complete!")
