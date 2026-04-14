import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


def load_and_preprocess_data(filepath):
    """Wczytuje dane, definiuje kolumny i przygotowuje macierze cech oraz używek."""
    columns = [
        "ID",
        "Age",
        "Gender",
        "Education",
        "Country",
        "Ethnicity",
        "Nscore",
        "Escore",
        "Oscore",
        "Ascore",
        "Cscore",
        "Impulsive",
        "SS",
        "Alcohol",
        "Amphet",
        "Amyl",
        "Benzos",
        "Caff",
        "Cannabis",
        "Choc",
        "Coke",
        "Crack",
        "Ecstasy",
        "Heroin",
        "Ketamine",
        "Legalh",
        "LSD",
        "Meth",
        "Mushrooms",
        "Nicotine",
        "Semer",
        "VSA",
    ]

    data = pd.read_csv(filepath, header=None, names=columns)

    features = columns[6:13]
    drug_columns = columns[13:]

    X = data[features]
    # Konwersja etykiet CL na wartości numeryczne
    Y = data[drug_columns].apply(lambda col: col.str.replace("CL", "").astype(int))

    return data, X, Y, features


def display_statistics(X):
    """Prezentuje statystyki opisowe, skośność oraz weryfikuje braki danych."""
    print("Statystyki opisowe:\n", X.describe())
    print("\nSkośność:\n", X.skew())
    print("\nBrakujące dane:\n", X.isnull().sum())


def plot_distributions(X, features):
    """Generuje histogramy oraz boxploty dla cech osobowości."""
    # Histogramy
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(features):
        axes[i].hist(X[col], bins=30)
        axes[i].set_title(col)
    axes[7].set_visible(False)
    plt.tight_layout()
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=X, orient="h")
    plt.title("Boxplot cech osobowości")
    plt.show()


def plot_correlation(X):
    """Generuje macierz korelacji cech osobowości."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Macierz korelacji cech osobowości")
    plt.show()


def plot_elbow_method(X, max_k=10):
    """Generuje wykres metody łokcia (inertia) dla k od 1 do max_k."""
    inertias = []
    k_range = range(1, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertias, "bo-")
    plt.xlabel("Liczba klastrów (k)")
    plt.ylabel("Inercja (Inertia)")
    plt.title("Metoda łokcia")
    plt.show()


def plot_silhouette_scores(X, max_k=10):
    """Generuje wykres Silhouette Score dla k od 2 do max_k."""
    scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, scores, "ro-")
    plt.xlabel("Liczba klastrów (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Wykres Silhouette Score")
    plt.show()


def perform_kmeans(X, k):
    """Przeprowadza klasteryzację K-Means dla zadanej liczby k i zwraca model."""
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    return model


def get_outliers_count(X, threshold=3):
    """Identyfikuje liczbę wierszy zawierających co najmniej jeden element odstający (z-score)."""
    z_scores = np.abs(zscore(X))
    outlier_mask = (z_scores > threshold).any(axis=1)
    count = outlier_mask.sum()
    print(f"Liczba wierszy z outlierami: {count}")
    return count


def run_hierarchical_clustering(X, k):
    """Przeprowadza klasteryzację hierarchiczną metodą Warda i rysuje dendrogram."""
    Z = linkage(X, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="level", p=k)
    plt.title("Dendrogram (obcięte drzewo)")
    plt.xlabel("Indeks próbki / poddrzewa")
    plt.ylabel("Odległość")
    plt.show()

    clusters = fcluster(Z, t=k, criterion="maxclust")
    return clusters


def visualize_pca(X, clusters, k, label):
    """Redukuje wymiarowość do 2D za pomocą PCA i wizualizuje przypisanie do skupień."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    print(f"PCA ({label}): PC1={var[0]:.2f}%, PC2={var[1]:.2f}%, suma={var.sum():.2f}%")

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", alpha=0.8)
    plt.title(f"PCA 2D - klasteryzacja {label} (k={k})")
    plt.xlabel(f"PC1 ({var[0]:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]:.1f}%)")
    plt.colorbar(scatter, label="Numer skupienia")
    plt.show()


def plot_cluster_profiles(X, clusters, features, label):
    """Tworzy heatmapę średnich wartości cech osobowości dla każdego skupienia."""
    df_clusters = X.copy()
    df_clusters["Cluster"] = clusters
    means = df_clusters.groupby("Cluster")[features].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Heatmapa profilu skupień {label} (średnie cech)")
    plt.xlabel("Cechy")
    plt.ylabel("Skupienie")
    plt.tight_layout()
    plt.show()


def plot_demographic_profile(data, clusters, label):
    """Wykresy słupkowe rozkładu Age, Education, Country per klaster."""
    age_map = {
        -0.95197: "18-24",
        -0.07854: "25-34",
        0.49788: "35-44",
        1.09449: "45-54",
        1.82213: "55-64",
        2.59171: "65+",
    }
    education_map = {
        -2.43591: "Porzucona szkoła",
        -1.73790: "Szkoła w trakcie",
        -1.43719: "Szkoła ukończona",
        -1.22751: "Studium zawodowe",
        -0.61113: "Uniwersytet w trakcie",
        -0.05921: "Uniwersytet ukończony",
        0.45468: "Magister",
        1.16365: "Doktorat w trakcie",
        1.98437: "Doktorat",
    }
    country_map = {
        -0.57009: "Australia",
        -0.46841: "Kanada",
        -0.28519: "Nowa Zelandia",
        -0.09765: "Inne",
        0.21128: "Irlandia",
        0.24923: "UK",
        0.96082: "USA",
    }
    label_maps = {"Age": age_map, "Education": education_map, "Country": country_map}

    df = data[["Age", "Education", "Country"]].copy()
    for col in ["Age", "Education", "Country"]:
        df[col] = df[col].map(label_maps[col])
    df["Cluster"] = clusters

    for col in ["Age", "Education", "Country"]:
        ct = pd.crosstab(df["Cluster"], df[col], normalize="index")
        print(f"\nRozkład {col} per klaster ({label}) [%]:")
        print((ct * 100).round(1).to_string())
        ct.plot(kind="bar", stacked=True, figsize=(10, 5))
        plt.title(f"Rozkład {col} per klaster ({label})")
        plt.xlabel("Skupienie")
        plt.ylabel("Udział")
        plt.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def plot_drug_profile(Y, clusters, label):
    """Heatmapa średnich wartości spożycia wybranych substancji per klaster."""
    selected_drugs = ["Cannabis", "Coke", "Alcohol", "Ecstasy", "Nicotine"]
    df = Y[selected_drugs].copy()
    df["Cluster"] = clusters
    means = df.groupby("Cluster")[selected_drugs].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(means, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title(f"Średnie spożycie substancji per klaster ({label})")
    plt.xlabel("Substancja")
    plt.ylabel("Skupienie")
    plt.tight_layout()
    plt.show()


def compare_clustering_methods(X, kmeans_labels, ward_labels):
    """Porównuje K-Means i Ward: Silhouette Score + Adjusted Rand Index."""
    sil_kmeans = silhouette_score(X, kmeans_labels)
    sil_ward = silhouette_score(X, ward_labels)
    ari = adjusted_rand_score(kmeans_labels, ward_labels)

    comparison = pd.DataFrame(
        {
            "Metryka": ["Silhouette Score", "Silhouette Score", "Adjusted Rand Index"],
            "Metoda": ["K-Means", "Ward", "K-Means vs Ward"],
            "Wartość": [sil_kmeans, sil_ward, ari],
        }
    )
    print("\nPorównanie metod klasteryzacji:")
    print(comparison.to_string(index=False))
    return comparison


if __name__ == "__main__":
    filepath = "drug_consumption.data"

    # 1. Przygotowanie danych
    data, X, Y, features = load_and_preprocess_data(filepath)

    # 2. Analiza wstępna (EDA)
    display_statistics(X)
    plot_distributions(X, features)
    plot_correlation(X)
    get_outliers_count(X)

    # 3. Klasteryzacja Hierarchiczna Warda
    h_labels = run_hierarchical_clustering(X, k=3)
    visualize_pca(X, h_labels, 3, "hierarchiczna")
    plot_cluster_profiles(X, h_labels, features, "hierarchiczna")

    # 4. Klasteryzacja K-Means
    plot_elbow_method(X, max_k=10)
    plot_silhouette_scores(X, max_k=10)
    kmeans_model = perform_kmeans(X, k=3)
    kmeans_labels = kmeans_model.labels_ + 1
    print("Rozmiary klastrów K-Means:", np.unique(kmeans_labels, return_counts=True))
    visualize_pca(X, kmeans_labels, 3, "K-Means")
    plot_cluster_profiles(X, kmeans_labels, features, "K-Means")

    # 5. Porównanie metod
    compare_clustering_methods(X, kmeans_labels, h_labels)

    # 6. Profil demograficzny i spożycia substancji
    plot_demographic_profile(data, kmeans_labels, "K-Means")
    plot_demographic_profile(data, h_labels, "Ward")
    plot_drug_profile(Y, kmeans_labels, "K-Means")
    plot_drug_profile(Y, h_labels, "Ward")