import React, { useState } from 'react';
import { AlertTriangle, ExternalLink, IndianRupee, ShieldCheck, Sparkles, X } from 'lucide-react';

function formatInr(value) {
  if (value === null || value === undefined || String(value).trim() === '') return null;
  const numeric = Number(String(value).replace(/[₹,\s]/g, ''));
  if (!Number.isFinite(numeric)) return String(value);
  return `₹${numeric.toLocaleString('en-IN', { maximumFractionDigits: 2 })}`;
}

function PriceBlock({ label, value, status, sourceUrl, date, note }) {
  const display = formatInr(value);
  return (
    <div className={`result-price-card ${display ? '' : 'unavailable'}`}>
      <small>{label}</small>
      <strong>{display || 'Unavailable'}</strong>
      {status && <span className="result-meta">{status}</span>}
      {date && <span className="result-meta">As of {date}</span>}
      {note && <span className="result-meta">{note}</span>}
      {sourceUrl && (
        <a href={sourceUrl} target="_blank" rel="noopener noreferrer">
          Source <ExternalLink size={12} />
        </a>
      )}
    </div>
  );
}

function MedicineCard({ medicine }) {
  const savings = medicine.net_savings;
  return (
    <article className="result-medicine">
      <header>
        <h3>{medicine.name || medicine.generic_name || 'Medicine result'}</h3>
        {medicine.match_source && <span className="result-chip">{medicine.match_source}</span>}
      </header>
      <dl>
        {medicine.generic_name && <><dt>Generic / ingredient</dt><dd>{medicine.generic_name}</dd></>}
        {medicine.brand_name && <><dt>Brand listing</dt><dd>{medicine.brand_name}</dd></>}
        {medicine.strength && <><dt>Strength</dt><dd>{medicine.strength}</dd></>}
        {medicine.dosage_form && <><dt>Dosage form</dt><dd>{medicine.dosage_form}</dd></>}
        {medicine.pack_size && <><dt>Pack</dt><dd>{medicine.pack_size}</dd></>}
        {medicine.verification_status && <><dt>Verification</dt><dd>{medicine.verification_status}</dd></>}
        {medicine.source && <><dt>Record source</dt><dd>{medicine.source}</dd></>}
      </dl>
      <div className="result-price-grid">
        <PriceBlock
          label="Jan Aushadhi price"
          value={medicine.jan_aushadhi_price}
          status={medicine.jan_price_status}
          sourceUrl={medicine.jan_price_source_url}
          date={medicine.jan_price_date}
        />
        <PriceBlock
          label="Branded MRP"
          value={medicine.branded_mrp}
          status={medicine.branded_mrp_status}
          sourceUrl={medicine.branded_mrp_source_url}
          date={medicine.branded_mrp_date}
        />
        <PriceBlock
          label="Observed market price"
          value={medicine.market_mrp}
          status={medicine.market_price_status}
          sourceUrl={medicine.market_price_source_url}
          date={medicine.market_price_date}
          note="Observed listing, not a substitute for a verified branded MRP."
        />
      </div>
      {savings && (
        <p className={`result-savings ${savings.available ? 'available' : ''}`}>
          <IndianRupee size={14} />
          {savings.available
            ? `${savings.message}: ${formatInr(savings.savings_per_pack)} (${savings.savings_percentage}% ${savings.currency})`
            : savings.message || 'Comparable savings are not available for this record.'}
        </p>
      )}
      <p className="result-caution">
        Matching ingredients, strength, and form does not mean a product is a confirmed substitute. Confirm with a doctor or pharmacist.
      </p>
    </article>
  );
}

export default function SearchResultModal({ open, loading, query, result, error, onClose, onClarifySearch }) {
  const [clarification, setClarification] = useState('');
  if (!open) return null;
  function submitClarification(event) {
    event.preventDefault();
    if (clarification.trim() && typeof onClarifySearch === 'function') {
      onClarifySearch(`${query} ${clarification.trim()}`);
      setClarification('');
    }
  }

  return (
    <div className="result-modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="result-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="search-result-title"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="result-modal-header">
          <div>
            <p className="result-kicker">Live medicine search</p>
            <h2 id="search-result-title">{query ? `Results for “${query}”` : 'Medicine search'}</h2>
          </div>
          <button type="button" className="result-close" onClick={onClose} aria-label="Close search results">
            <X size={18} />
          </button>
        </header>

        {loading && <p className="result-status">Searching live medicine sources…</p>}

        {!loading && error && (
          <div className="result-alert error" role="alert">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        {!loading && result?.status === 'error' && (
          <div className="result-alert error" role="alert">
            <AlertTriangle size={18} />
            <span>{result.message || 'Search could not be completed.'}</span>
          </div>
        )}

        {!loading && result?.status === 'not_found' && (
          <div className="result-alert" role="status">
            <ShieldCheck size={18} />
            <span>{result.message || 'No reliable live source match found. DawaWise will not invent a medicine record.'}</span>
          </div>
        )}

        {!loading && result?.status === 'matched' && result.medicine && (
          <>
            {result.match_source === 'openrouter' && (
              <div className="result-alert warn" role="status">
                <Sparkles size={18} />
                <span>
                  This listing was identified by the search planner and checked against retrieved source content.
                  {result.requires_confirmation ? ' Please confirm it is the intended product.' : ''}
                  {result.reason ? ` ${result.reason}` : ''}
                </span>
              </div>
            )}
            <MedicineCard medicine={{ ...result.medicine, match_source: result.match_source }} />
          </>
        )}

        {!loading && result?.status === 'clarification_required' && (
          <>
            <div className="result-alert warn" role="status">
              <AlertTriangle size={18} />
              <span>{result.question || 'Please add a brand or dosage-form detail to refine the live search.'}</span>
            </div>
            <form className="result-clarification-form" onSubmit={submitClarification}>
              <label className="sr-only" htmlFor="result-clarification">Add a brand or dosage-form detail</label>
              <input id="result-clarification" value={clarification} onChange={(event) => setClarification(event.target.value)} placeholder="Brand or dosage form" required />
              <button type="submit" disabled={!clarification.trim() || !onClarifySearch}>Search</button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
