# ── GKE node service account ─────────────────────────────────────────────────
# Nodes use this SA — minimal permissions, not the default compute SA
resource "google_service_account" "gke_node" {
  account_id   = "${var.cluster_name}-node-sa"
  display_name = "GKE Node Service Account"
  project      = var.project_id
}

resource "google_project_iam_member" "node_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.gke_node.email}"
}

resource "google_project_iam_member" "node_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.gke_node.email}"
}

# ── Triton Workload Identity ──────────────────────────────────────────────────
# Allows the Triton k8s ServiceAccount to impersonate this GCP SA
# and read models from GCS — no static keys needed
resource "google_service_account" "triton" {
  account_id   = "${var.cluster_name}-triton-sa"
  display_name = "Triton Inference Server Service Account"
  project      = var.project_id
}

# ── GitHub Actions OIDC ───────────────────────────────────────────────────────
# GitHub Actions federates identity with GCP — no stored credentials anywhere
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "gh-pool-v2"
  display_name              = "GitHub Actions Pool"
  project                   = var.project_id
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "gh-provider-v2"
  project                            = var.project_id

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  attribute_condition = "assertion.repository == '${var.github_repo}'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account" "github_actions" {
  account_id   = "${var.cluster_name}-github-sa"
  display_name = "GitHub Actions Deploy Service Account"
  project      = var.project_id
}

resource "google_service_account_iam_member" "github_oidc" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repo}"
}

resource "google_project_iam_member" "github_container_dev" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = "serviceAccount:${google_service_account.github_actions.email}"
}
