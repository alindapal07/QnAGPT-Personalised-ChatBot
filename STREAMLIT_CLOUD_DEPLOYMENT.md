# QnAGPT — Streamlit Community Cloud Deployment Guide

This guide provides complete, production-tested instructions for deploying and operating the **QnAGPT — Premium AI Workspace** on **Streamlit Community Cloud** (or equivalent containerized PaaS platforms).

---

## 1. Architecture & Feasibility Overview

Streamlit Community Cloud runs applications inside isolated container sandboxes. The key deployment characteristics are:

- **Single Process Python Runtime:** The application boots directly from `qna_bot/app.py`.
- **Zero Frontend Hosting Overhead:** No external Node.js, React server, or FastAPI gateway is needed.
- **Ephemeral Filesystem Lifecycle:** Local container disk storage is non-durable across application restarts, container sleep cycles, and redeployments.
- **Production Persistence Strategy:**
  - **Vector Storage:** Qdrant Cloud (Managed cluster with permanent vector indexing).
  - **Relational Storage:** Managed PostgreSQL or MySQL via `DATABASE_URL` (or local SQLite for single-session ephemeral demos).
  - **Document Storage:** Remote object store or session-staged ingestion into Qdrant.

---

## 2. GitHub Repository Preparation

Ensure your repository contains the following deployment assets:

1. **Main Entrypoint:** `qna_bot/app.py`
2. **Dependency Specification:** `requirements.txt` (clean, tested set of runtime dependencies)
3. **Python Runtime:** `runtime.txt` specifying `python-3.12.8`
4. **Server Configuration:** `.streamlit/config.toml` (XSRF enabled, CORS secured, loopback binding, upload ceilings)
5. **Secrets Template:** `.streamlit/secrets.toml.example`

Verify your `.gitignore` contains:
```gitignore
.streamlit/secrets.toml
*.db
.env
```
> **Security Rule:** Never commit real secrets or `.streamlit/secrets.toml` to Git.

---

## 3. Main Application File

In the Streamlit Cloud deployment dialog, configure:
- **Repository:** `alindapal07/QnAGPT-Personalised-ChatBot`
- **Branch:** `main` (or active development branch)
- **Main file path:** `qna_bot/app.py`

---

## 4. Python Runtime & Dependencies

Streamlit Cloud reads:
- `runtime.txt`: Pinning the container to Python `3.12.8`.
- `requirements.txt`: Streamlit automatically installs packages using `pip install -r requirements.txt`.

Our `requirements.txt` has been audited and cleaned in Phase 7.7:
- Unused packages (`langchain-openai`, `duckduckgo-search`) removed.
- Direct runtime dependencies (`langchain-core>=1.6.6`) explicitly declared.
- Heavy notebook tooling isolated to dev groups.

---

## 5. Secrets Management Setup

In the Streamlit Cloud dashboard:
1. Navigate to your app dashboard.
2. Click **Settings** (gear icon) -> **Secrets**.
3. Paste your production environment configuration (see `.streamlit/secrets.toml.example`):

```toml
# =============================================================================
# QnAGPT Streamlit Community Cloud Production Secrets
# =============================================================================

# Application Mode
APP_ENV = "production"

# Primary LLM API Key (Required)
GROQ_API_KEY = "gsk_your_actual_groq_api_key_here"

# HMAC Session Signing Key (Required: minimum 32 chars in production)
# Generate via: python -c "import secrets; print(secrets.token_hex(32))"
SESSION_SECRET_KEY = "your_64_character_cryptographically_secure_hex_key_here"

# Qdrant Cloud Vector Database (Recommended for permanent vector persistence)
QDRANT_MODE = "cloud"
QDRANT_URL = "https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "your_qdrant_cloud_api_key_here"
QDRANT_COLLECTION = "qna_documents"

# External Relational Database (Recommended for multi-user production)
DATABASE_URL = "postgresql://qnagpt_user:StrongPassword@db-host.provider.com:5432/qnagpt_prod"

# Google OAuth / OpenID Connect (Optional)
GOOGLE_CLIENT_ID = "your_google_client_id.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "GOCSPX-your_google_client_secret"

# GitHub OAuth (Optional)
GITHUB_CLIENT_ID = "your_github_client_id"
GITHUB_CLIENT_SECRET = "your_github_client_secret"
```

---

## 6. External Database Configuration

Because Streamlit Cloud containers restart and sleep after inactivity:
1. **Local SQLite Limitation:** If `DATABASE_URL` is omitted, the app defaults to `sqlite:///chatbot.db`. On container rebuilds or idle sleep waking, local SQLite files may reset.
2. **Production Multi-User Persistence:** Use a managed PostgreSQL database (e.g. Supabase, Neon, AWS RDS, GCP Cloud SQL, or Railway).
3. Set `DATABASE_URL` in Streamlit Cloud secrets:
   `postgresql://user:password@host:port/dbname`
4. The SQLAlchemy engine in `qna_bot/database/session.py` automatically initializes schemas on startup.

---

## 7. Qdrant Cloud Vector Store Configuration

To persist document embeddings across container rebuilds:
1. Create a free cluster on [Qdrant Cloud](https://cloud.qdrant.io/).
2. Obtain your **Cluster URL** (e.g., `https://xyz.us-east4-0.gcp.cloud.qdrant.io:6333`) and **API Key**.
3. In Streamlit Cloud Secrets, specify:
   ```toml
   QDRANT_MODE = "cloud"
   QDRANT_URL = "https://xyz.us-east4-0.gcp.cloud.qdrant.io:6333"
   QDRANT_API_KEY = "..."
   QDRANT_COLLECTION = "qna_documents"
   ```
4. Qdrant collections will be automatically provisioned with 384-dimensional cosine index on first document ingestion. All vectors are strictly segmented by `user_id` payload filters.

---

## 8. OAuth Callback Configuration

For Google and GitHub OAuth authentication:
1. Determine your Streamlit Cloud app URL:
   `https://<your-app-subdomain>.streamlit.app`
2. In **Google Cloud Console** -> **Credentials** -> **OAuth 2.0 Client IDs**:
   - Authorized JavaScript Origins: `https://<your-app-subdomain>.streamlit.app`
   - Authorized Redirect URIs: `https://<your-app-subdomain>.streamlit.app`
3. In **GitHub Developer Settings** -> **OAuth Apps**:
   - Homepage URL: `https://<your-app-subdomain>.streamlit.app`
   - Authorization callback URL: `https://<your-app-subdomain>.streamlit.app`
4. The application automatically handles the `?oauth_provider=google&code=...&state=...` query parameters upon redirection and completes cryptographic nonce & PKCE validation.

---

## 9. Streamlit Community Cloud Step-by-Step Deployment

1. Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
2. Click **New app**.
3. Select your repository: `alindapal07/QnAGPT-Personalised-ChatBot`.
4. Select the Branch: `main`.
5. Enter Main file path: `qna_bot/app.py`.
6. Expand **Advanced settings**:
   - Python version: `3.12`.
   - Secrets: Paste the configured secrets from Section 5.
7. Click **Deploy!**.

---

## 10. First Deployment Validation & Acceptance Checklist

Once the container boots:
1. **Application Shell:** Verify the dual-card layout appears against the `#EEF3FB` background.
2. **Authentication Gate:**
   - Verify the login/registration form renders.
   - Register a test user (e.g., `admin@yourdomain.com`).
   - Log in and verify session token creation.
3. **Chat Workspace:**
   - Verify empty state heading: *"What would you like to explore today?"*.
   - Click a suggestion card (e.g., *"Explain a technical concept"*).
   - Verify response streaming / synthesis from Groq LLM.
   - Verify the floating composer sends messages properly.
4. **Knowledge Base:**
   - Navigate to **Knowledge Base**.
   - Upload a test PDF / TXT document.
   - Verify chunking and vector indexing into Qdrant.
   - Switch back to Chat and ask a question referencing the uploaded document.
   - Confirm document citations appear in the sources panel.
5. **Approval Center:**
   - Navigate to **Approvals** to inspect tool authorization records.
6. **Sign Out:**
   - Sign out and confirm session token invalidation.

---

## 11. Troubleshooting Common Cloud Deployment Issues

### Issue 1: App Fails to Boot with `ConfigurationError: Production environment requires an explicit SESSION_SECRET_KEY`
- **Cause:** `APP_ENV = "production"` is set in secrets, but `SESSION_SECRET_KEY` is missing or under 32 characters.
- **Fix:** Generate a 64-character secret using `python -c "import secrets; print(secrets.token_hex(32))"` and paste it into Streamlit Cloud secrets.

### Issue 2: Qdrant Connection Refused
- **Cause:** `QDRANT_MODE` is left as `"local"` which tries to contact `localhost:6333` inside the cloud container.
- **Fix:** Set `QDRANT_MODE = "cloud"` and provide valid `QDRANT_URL` and `QDRANT_API_KEY`.

### Issue 3: OAuth Login Fails with "Mismatched Nonce" or "Invalid Redirect URI"
- **Cause:** The Google / GitHub OAuth console does not have the exact `https://<your-app>.streamlit.app` URL registered in Authorized Redirect URIs.
- **Fix:** Update the authorized redirect URI in Google/GitHub console to match the exact live Streamlit domain.

### Issue 4: Out-Of-Memory (OOM) on Large Document Uploads
- **Cause:** Streamlit Cloud free tier provides 1GB RAM. Processing 100MB+ documents in-memory can exceed limits.
- **Fix:** Upload documents within the 20MB limit (governed by `maxUploadSize = 25` in `.streamlit/config.toml`).

---

## 12. Backup, Recovery & Maintenance

1. **Relational Database Backups:** Configure daily automated snapshots on your managed PostgreSQL provider.
2. **Qdrant Snapshots:** Utilize Qdrant Cloud's built-in collection snapshot utility.
3. **Zero Downtime Updates:** Pushing commits to the connected GitHub branch automatically triggers a rolling redeployment on Streamlit Community Cloud.
