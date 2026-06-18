# GKE Standard cluster — explicit node pools, full control over machine types.
# Standard vs Autopilot: Standard lets us configure GPU node pools explicitly,
resource "google_container_cluster" "primary" {
  name                = var.cluster_name
  location            = var.region
  project             = var.project_id
  deletion_protection = false

  # Remove the default node pool — we create explicit pools below
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = var.network
  subnetwork = var.subnetwork

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Workload Identity — pods authenticate to GCP APIs without static keys
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }
}

# ── CPU node pool — system pods, gateway ─────────────────────────────────────
resource "google_container_node_pool" "cpu_nodes" {
  name     = "cpu-pool"
  cluster  = google_container_cluster.primary.id
  location = var.region
  project  = var.project_id

  initial_node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 3
  }

  node_config {
    machine_type    = "e2-standard-2" # 2 vCPU, 8GB RAM
    service_account = var.node_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    labels = {
      pool = "cpu"
    }
  }
}

resource "google_container_node_pool" "gpu_nodes" {
  name     = "gpu-pool"
  cluster  = google_container_cluster.primary.id
  location = var.region    # regional, not zonal
  project  = var.project_id

  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = 2
    location_policy = "ANY"
  }

  node_config {
    machine_type    = "n1-standard-4"    # T4 GPU machine
    service_account = var.node_service_account
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "DEFAULT"
      }
    }

    taint {
      key    = "nvidia.com/gpu"
      value  = "present"
      effect = "NO_SCHEDULE"
    }

    labels = {
      pool             = "gpu"
      accelerator-type = "nvidia-tesla-t4"
    }
  }
}
