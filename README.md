# Kubernetes GitOps Manifests

Portfolio sample of the GitOps patterns I run on EKS: an ArgoCD
app-of-apps structure, a Helm values file with production hardening
(resource limits, PDB, HPA, pod security), and a default-deny
NetworkPolicy template. Directory per app, environment overlays via
Kustomize-style value files.

## Layout

- `argocd/root-app.yaml` — app-of-apps that syncs every child app
- `apps/web-api/values.yaml` — hardened Helm values for a sample service
- `apps/web-api/templates/networkpolicy.yaml` — default-deny + allowlist egress
- `apps/web-api/templates/pdb.yaml` — PodDisruptionBudget

## Patterns demonstrated

- App-of-apps with automated sync, prune, and self-heal
- Resource requests/limits on every container (no BestEffort in prod)
- PDBs sized for rolling updates (maxUnavailable: 1)
- Default-deny NetworkPolicies with explicit allowlists
- Read-only root filesystem + non-root users in pod security context
