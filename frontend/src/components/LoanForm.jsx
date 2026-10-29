import { useState, useEffect } from 'react'

export const SAMPLE_APPROVED = {
  no_of_dependents: 2,
  education: 'Graduate',
  self_employed: 'No',
  income_annum: 9600000,
  loan_amount: 29900000,
  loan_term: 12,
  cibil_score: 778,
  residential_assets_value: 2400000,
  commercial_assets_value: 17600000,
  luxury_assets_value: 22700000,
  bank_asset_value: 8000000,
}

export const SAMPLE_REJECTED = {
  no_of_dependents: 0,
  education: 'Not Graduate',
  self_employed: 'Yes',
  income_annum: 4100000,
  loan_amount: 12200000,
  loan_term: 8,
  cibil_score: 417,
  residential_assets_value: 2700000,
  commercial_assets_value: 2200000,
  luxury_assets_value: 8800000,
  bank_asset_value: 3300000,
}

export const INITIAL_FORM = {
  no_of_dependents: '',
  education: 'Graduate',
  self_employed: 'No',
  income_annum: '',
  loan_amount: '',
  loan_term: '',
  cibil_score: '',
  residential_assets_value: '',
  commercial_assets_value: '',
  luxury_assets_value: '',
  bank_asset_value: '',
}

export const formatINR = (val) => {
  if (val === '' || val == null) return ''
  const num = Number(val)
  if (isNaN(num)) return ''
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(num)
}

export const formatINRWords = (val) => {
  if (val === '' || val == null) return ''
  const num = Number(val)
  if (isNaN(num) || num <= 0) return ''
  if (num >= 1e7) {
    return `${(num / 1e7).toFixed(2)} Crore`
  }
  if (num >= 1e5) {
    return `${(num / 1e5).toFixed(2)} Lakh`
  }
  return formatINR(num)
}

export default function LoanForm({ onSubmit, loading, externalForm, onReset }) {
  const [form, setForm] = useState(INITIAL_FORM)
  const [errors, setErrors] = useState({})

  // Sync external sample changes if provided
  useEffect(() => {
    if (externalForm) {
      setForm({ ...externalForm })
      setErrors({})
    }
  }, [externalForm])

  const setField = (field, raw) => {
    setForm(prev => ({ ...prev, [field]: raw }))
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }))
    }
  }

  const validate = () => {
    const errs = {}
    const numFields = [
      ['no_of_dependents', 0, 20, 'Dependents'],
      ['income_annum', 1, 1e12, 'Annual Income'],
      ['loan_amount', 1, 1e12, 'Loan Amount'],
      ['loan_term', 1, 50, 'Loan Term'],
      ['cibil_score', 300, 900, 'CIBIL Score'],
      ['residential_assets_value', 0, 1e12, 'Residential Assets'],
      ['commercial_assets_value', 0, 1e12, 'Commercial Assets'],
      ['luxury_assets_value', 0, 1e12, 'Luxury Assets'],
      ['bank_asset_value', 0, 1e12, 'Bank Assets'],
    ]

    numFields.forEach(([f, min, max, label]) => {
      const v = Number(form[f])
      if (form[f] === '' || isNaN(v)) {
        errs[f] = `${label} is required`
      } else if (v < min || v > max) {
        errs[f] = `${label} must be ${min}-${max}`
      }
    })

    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!validate()) return

    const payload = { ...form }
    // Convert numeric fields to Number
    Object.keys(payload).forEach(k => {
      if (!['education', 'self_employed'].includes(k)) {
        payload[k] = Number(payload[k])
      }
    })
    onSubmit(payload)
  }

  const fillSample = (sample) => {
    setForm({ ...sample })
    setErrors({})
  }

  const handleClear = () => {
    setForm(INITIAL_FORM)
    setErrors({})
    if (onReset) onReset()
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="card-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
              <polyline points="10 9 9 9 8 9"></polyline>
            </svg>
          </div>
          <div>
            <h2 className="card-title">Applicant Assessment Form</h2>
            <p className="card-subtitle">Provide verified applicant profile data for real-time model scoring</p>
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} noValidate>
        {/* Section 1: Applicant Details */}
        <div className="form-section">
          <div className="section-header">
            <span className="section-number">1</span>
            <span className="section-title">Applicant Details</span>
          </div>
          <div className="form-grid-2">
            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="no_of_dependents">
                  Number of Dependents
                </label>
                <span className="form-unit-tag">Range: 0-20</span>
              </div>
              <input
                id="no_of_dependents"
                type="number"
                className="form-input"
                value={form.no_of_dependents}
                onChange={e => setField('no_of_dependents', e.target.value)}
                placeholder="e.g. 2"
                min="0"
                max="20"
                step="1"
                aria-invalid={!!errors.no_of_dependents}
                aria-describedby={errors.no_of_dependents ? 'err-dependents' : undefined}
              />
              {errors.no_of_dependents && (
                <span id="err-dependents" className="form-error-text">{errors.no_of_dependents}</span>
              )}
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="education">
                  Education Level
                </label>
              </div>
              <select
                id="education"
                className="form-select"
                value={form.education}
                onChange={e => setField('education', e.target.value)}
              >
                <option value="Graduate">Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
              </select>
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="self_employed">
                  Employment Status
                </label>
              </div>
              <select
                id="self_employed"
                className="form-select"
                value={form.self_employed}
                onChange={e => setField('self_employed', e.target.value)}
              >
                <option value="No">Salaried / Not Self-Employed</option>
                <option value="Yes">Self-Employed</option>
              </select>
            </div>
          </div>
        </div>

        {/* Section 2: Loan Details */}
        <div className="form-section">
          <div className="section-header">
            <span className="section-number">2</span>
            <span className="section-title">Loan Details</span>
          </div>
          <div className="form-grid-2">
            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="loan_amount">
                  Loan Amount
                </label>
                <span className="form-unit-tag">INR (₹)</span>
              </div>
              <input
                id="loan_amount"
                type="number"
                className="form-input"
                value={form.loan_amount}
                onChange={e => setField('loan_amount', e.target.value)}
                placeholder="e.g. 10000000"
                min="1"
                aria-invalid={!!errors.loan_amount}
              />
              <div className="form-helper-text">
                {form.loan_amount && !errors.loan_amount ? (
                  <span className="currency-preview">
                    {formatINR(form.loan_amount)} {formatINRWords(form.loan_amount) ? `(${formatINRWords(form.loan_amount)})` : ''}
                  </span>
                ) : (
                  <span>Principal request amount</span>
                )}
              </div>
              {errors.loan_amount && (
                <span className="form-error-text">{errors.loan_amount}</span>
              )}
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="loan_term">
                  Loan Term
                </label>
                <span className="form-unit-tag">Years (1-50)</span>
              </div>
              <input
                id="loan_term"
                type="number"
                className="form-input"
                value={form.loan_term}
                onChange={e => setField('loan_term', e.target.value)}
                placeholder="e.g. 12"
                min="1"
                max="50"
                step="1"
                aria-invalid={!!errors.loan_term}
              />
              <div className="form-helper-text">
                <span>Repayment tenure in years</span>
              </div>
              {errors.loan_term && (
                <span className="form-error-text">{errors.loan_term}</span>
              )}
            </div>
          </div>
        </div>

        {/* Section 3: Income and Assets */}
        <div className="form-section">
          <div className="section-header">
            <span className="section-number">3</span>
            <span className="section-title">Income & Asset Holdings</span>
          </div>
          <div className="form-grid-2">
            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="income_annum">
                  Annual Income
                </label>
                <span className="form-unit-tag">INR (₹)</span>
              </div>
              <input
                id="income_annum"
                type="number"
                className="form-input"
                value={form.income_annum}
                onChange={e => setField('income_annum', e.target.value)}
                placeholder="e.g. 5000000"
                min="0"
                aria-invalid={!!errors.income_annum}
              />
              <div className="form-helper-text">
                {form.income_annum && !errors.income_annum ? (
                  <span className="currency-preview">
                    {formatINR(form.income_annum)} {formatINRWords(form.income_annum) ? `(${formatINRWords(form.income_annum)})` : ''}
                  </span>
                ) : (
                  <span>Gross annual income</span>
                )}
              </div>
              {errors.income_annum && (
                <span className="form-error-text">{errors.income_annum}</span>
              )}
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="residential_assets_value">
                  Residential Assets
                </label>
                <span className="form-unit-tag">INR (₹)</span>
              </div>
              <input
                id="residential_assets_value"
                type="number"
                className="form-input"
                value={form.residential_assets_value}
                onChange={e => setField('residential_assets_value', e.target.value)}
                placeholder="e.g. 2400000"
                min="0"
                aria-invalid={!!errors.residential_assets_value}
              />
              <div className="form-helper-text">
                {form.residential_assets_value && !errors.residential_assets_value ? (
                  <span className="currency-preview">
                    {formatINR(form.residential_assets_value)} {formatINRWords(form.residential_assets_value) ? `(${formatINRWords(form.residential_assets_value)})` : ''}
                  </span>
                ) : (
                  <span>Appraised home/land value</span>
                )}
              </div>
              {errors.residential_assets_value && (
                <span className="form-error-text">{errors.residential_assets_value}</span>
              )}
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="commercial_assets_value">
                  Commercial Assets
                </label>
                <span className="form-unit-tag">INR (₹)</span>
              </div>
              <input
                id="commercial_assets_value"
                type="number"
                className="form-input"
                value={form.commercial_assets_value}
                onChange={e => setField('commercial_assets_value', e.target.value)}
                placeholder="e.g. 17600000"
                min="0"
                aria-invalid={!!errors.commercial_assets_value}
              />
              <div className="form-helper-text">
                {form.commercial_assets_value && !errors.commercial_assets_value ? (
                  <span className="currency-preview">
                    {formatINR(form.commercial_assets_value)} {formatINRWords(form.commercial_assets_value) ? `(${formatINRWords(form.commercial_assets_value)})` : ''}
                  </span>
                ) : (
                  <span>Commercial property/office value</span>
                )}
              </div>
              {errors.commercial_assets_value && (
                <span className="form-error-text">{errors.commercial_assets_value}</span>
              )}
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="luxury_assets_value">
                  Luxury Assets
                </label>
                <span className="form-unit-tag">INR (₹)</span>
              </div>
              <input
                id="luxury_assets_value"
                type="number"
                className="form-input"
                value={form.luxury_assets_value}
                onChange={e => setField('luxury_assets_value', e.target.value)}
                placeholder="e.g. 22700000"
                min="0"
                aria-invalid={!!errors.luxury_assets_value}
              />
              <div className="form-helper-text">
                {form.luxury_assets_value && !errors.luxury_assets_value ? (
                  <span className="currency-preview">
                    {formatINR(form.luxury_assets_value)} {formatINRWords(form.luxury_assets_value) ? `(${formatINRWords(form.luxury_assets_value)})` : ''}
                  </span>
                ) : (
                  <span>Vehicles, jewelry, collectibles</span>
                )}
              </div>
              {errors.luxury_assets_value && (
                <span className="form-error-text">{errors.luxury_assets_value}</span>
              )}
            </div>

            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="bank_asset_value">
                  Bank Deposits / Liquid Assets
                </label>
                <span className="form-unit-tag">INR (₹)</span>
              </div>
              <input
                id="bank_asset_value"
                type="number"
                className="form-input"
                value={form.bank_asset_value}
                onChange={e => setField('bank_asset_value', e.target.value)}
                placeholder="e.g. 8000000"
                min="0"
                aria-invalid={!!errors.bank_asset_value}
              />
              <div className="form-helper-text">
                {form.bank_asset_value && !errors.bank_asset_value ? (
                  <span className="currency-preview">
                    {formatINR(form.bank_asset_value)} {formatINRWords(form.bank_asset_value) ? `(${formatINRWords(form.bank_asset_value)})` : ''}
                  </span>
                ) : (
                  <span>Savings, FDs, mutual funds</span>
                )}
              </div>
              {errors.bank_asset_value && (
                <span className="form-error-text">{errors.bank_asset_value}</span>
              )}
            </div>
          </div>
        </div>

        {/* Section 4: Credit and Financial History */}
        <div className="form-section">
          <div className="section-header">
            <span className="section-number">4</span>
            <span className="section-title">Credit & Financial History</span>
          </div>
          <div className="form-grid-1">
            <div className="form-group">
              <div className="form-label-row">
                <label className="form-label" htmlFor="cibil_score">
                  CIBIL Credit Score
                </label>
                <span className="form-unit-tag">Valid Range: 300 - 900</span>
              </div>
              <input
                id="cibil_score"
                type="number"
                className="form-input"
                value={form.cibil_score}
                onChange={e => setField('cibil_score', e.target.value)}
                placeholder="e.g. 778"
                min="300"
                max="900"
                step="1"
                aria-invalid={!!errors.cibil_score}
              />
              <div className="form-helper-text">
                <span>Credit bureau score (300-549: High Risk, 550-649: Subprime, 650-749: Near Prime, 750+: Prime)</span>
              </div>
              {errors.cibil_score && (
                <span className="form-error-text">{errors.cibil_score}</span>
              )}
            </div>
          </div>
        </div>

        {/* Form Actions */}
        <div className="form-actions">
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            id="submit-loan-btn"
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 18, height: 18, margin: 0, borderWidth: 2 }} />
                <span>Evaluating Credit Risk...</span>
              </>
            ) : (
              <>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                  <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                <span>Assess Application</span>
              </>
            )}
          </button>

          <div className="btn-group-row">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => fillSample(SAMPLE_APPROVED)}
              disabled={loading}
              title="Populate prime applicant data likely to be approved"
            >
              Fill Likely Approved
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => fillSample(SAMPLE_REJECTED)}
              disabled={loading}
              title="Populate subprime applicant data likely to be rejected"
            >
              Fill Likely Rejected
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={handleClear}
              disabled={loading}
            >
              Reset Form
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}
