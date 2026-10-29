import { useState } from 'react'
import axios from 'axios'
import './index.css'
import LoanForm, { SAMPLE_APPROVED, SAMPLE_REJECTED } from './components/LoanForm'
import PredictionResult from './components/PredictionResult'
import FeatureImportance from './components/FeatureImportance'
import ModelComparison from './components/ModelComparison'

const API_BASE = import.meta.env.VITE_API_BASE || ''

function App() {
  const [activeTab, setActiveTab] = useState('assess')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)
  const [externalForm, setExternalForm] = useState(null)
  const [lastSubmittedPayload, setLastSubmittedPayload] = useState(null)

  const handleSubmit = async (formData) => {
    setLoading(true)
    setError(null)
    setLastSubmittedPayload(formData)
    try {
      const resp = await axios.post(`${API_BASE}/api/v1/loan/explain`, formData)
      setResult(resp.data)
    } catch (err) {
      const msg = err.response?.data?.details
        ? err.response.data.details.join('; ')
        : err.response?.data?.error || err.message
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleRetry = () => {
    if (lastSubmittedPayload) {
      handleSubmit(lastSubmittedPayload)
    }
  }

  const handleLoadSample = (sample) => {
    setExternalForm({ ...sample })
    setResult(null)
    setError(null)
  }

  const handleNewAssessment = () => {
    setResult(null)
    setError(null)
  }

  return (
    <>
      {/* Clean Top Navigation Bar */}
      <nav className="navbar" role="navigation" aria-label="Main Navigation">
        <div className="navbar-inner">
          <div className="navbar-brand" onClick={() => setActiveTab('assess')}>
            <div className="brand-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
              </svg>
            </div>
            <div className="brand-info">
              <div className="brand-title">
                <span>Loan Assessment</span>
                <span className="brand-badge">ML Dashboard</span>
              </div>
              <span className="brand-subtitle">AI-Powered Credit Decisioning & Explainability</span>
            </div>
          </div>

          <div className="navbar-nav">
            <button
              className={`nav-item ${activeTab === 'assess' ? 'active' : ''}`}
              onClick={() => setActiveTab('assess')}
              id="nav-tab-assess"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
              <span>Assess</span>
            </button>
            <button
              className={`nav-item ${activeTab === 'models' ? 'active' : ''}`}
              onClick={() => setActiveTab('models')}
              id="nav-tab-models"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
              </svg>
              <span>Models</span>
            </button>
          </div>

          <div className="navbar-status">
            <span className="status-dot"></span>
            <span>Production: Random Forest &bull; Online</span>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="app-container">
        {activeTab === 'assess' && (
          <>
            <div className="page-header">
              <div>
                <h1 className="page-header-title">Loan Underwriting & Risk Assessment</h1>
                <p className="page-header-desc">
                  Real-time algorithmic credit appraisal with TreeSHAP feature contribution analysis
                </p>
              </div>
            </div>

            {/* Two-Column Dashboard Layout */}
            <div className="dashboard-grid">
              {/* Left Panel: Applicant Form */}
              <div className="form-panel-wrapper">
                <LoanForm
                  onSubmit={handleSubmit}
                  loading={loading}
                  externalForm={externalForm}
                  onReset={handleNewAssessment}
                />
              </div>

              {/* Right Panel: Assessment Results */}
              <div className="results-panel-wrapper">
                {/* State 1: Error State */}
                {error && !loading && (
                  <div className="error-card">
                    <div className="error-icon">&#9888;</div>
                    <div className="error-content">
                      <div className="error-title">Assessment Error</div>
                      <div className="error-msg">{error}</div>
                      <div style={{ marginTop: 10 }}>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={handleRetry}
                          style={{ padding: '6px 14px', fontSize: '0.8125rem' }}
                        >
                          Retry Assessment
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* State 2: Loading State */}
                {loading && (
                  <div className="card loading-card">
                    <div className="spinner" />
                    <h3 className="loading-title">Evaluating Credit Risk</h3>
                    <p className="loading-desc">
                      Executing Random Forest classification pipeline, computing interaction terms, and generating local TreeSHAP feature attributions...
                    </p>
                  </div>
                )}

                {/* State 3: Empty State (Before assessment) */}
                {!loading && !result && (
                  <div className="empty-state-card">
                    <div className="empty-state-icon-box">
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 14 14"></polyline>
                      </svg>
                    </div>

                    <h3 className="empty-state-title">Ready for Assessment</h3>
                    <p className="empty-state-desc">
                      Enter applicant details on the left or load a pre-configured sample profile to execute machine learning prediction and explainability analysis.
                    </p>

                    {/* Quick Load Sample Applicant Cards */}
                    <div className="sample-applicants-card">
                      <div className="sample-applicants-header">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                        </svg>
                        Quick Test Profiles
                      </div>
                      <div className="sample-profiles-grid">
                        <button
                          type="button"
                          className="sample-profile-btn"
                          onClick={() => handleLoadSample(SAMPLE_APPROVED)}
                        >
                          <div className="sample-profile-badge approved">Likely Approved</div>
                          <div className="sample-profile-title">Prime Applicant</div>
                          <div className="sample-profile-meta">CIBIL 778 &bull; ₹96L Income &bull; Graduate</div>
                        </button>

                        <button
                          type="button"
                          className="sample-profile-btn"
                          onClick={() => handleLoadSample(SAMPLE_REJECTED)}
                        >
                          <div className="sample-profile-badge rejected">Likely Rejected</div>
                          <div className="sample-profile-title">Subprime Applicant</div>
                          <div className="sample-profile-meta">CIBIL 417 &bull; ₹41L Income &bull; ₹1.22 Cr Loan</div>
                        </button>
                      </div>
                    </div>

                    {/* Pipeline Features Overview */}
                    <div className="pipeline-features-list">
                      <div className="pipeline-feature-item">
                        <span className="pipeline-feature-icon">&#10003;</span>
                        <span>Calibrated decision threshold for balanced precision & recall</span>
                      </div>
                      <div className="pipeline-feature-item">
                        <span className="pipeline-feature-icon">&#10003;</span>
                        <span>Local SHAP feature waterfall attribution (positive & risk factors)</span>
                      </div>
                      <div className="pipeline-feature-item">
                        <span className="pipeline-feature-icon">&#10003;</span>
                        <span>Automated FCRA-compliant Adverse-Action explanation summary</span>
                      </div>
                      <div className="pipeline-feature-item">
                        <span className="pipeline-feature-icon">&#10003;</span>
                        <span>Personalized action items to improve approval probability</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* State 4: Assessment Results View */}
                {result && !loading && (
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <PredictionResult
                      result={result}
                      onNewAssessment={handleNewAssessment}
                    />
                    <FeatureImportance result={result} />
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Models Page Tab */}
        {activeTab === 'models' && (
          <>
            <div className="page-header">
              <div>
                <h1 className="page-header-title">ML Models & Evaluation Benchmarks</h1>
                <p className="page-header-desc">
                  Performance comparison across candidate scikit-learn pipelines on the holdout test dataset
                </p>
              </div>
            </div>
            <ModelComparison apiBase={API_BASE} />
          </>
        )}
      </main>
    </>
  )
}

export default App
