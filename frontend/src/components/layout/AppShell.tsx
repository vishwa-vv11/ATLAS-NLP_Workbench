import type { ReactNode } from "react";
import { BookOpen, Info, Menu } from "lucide-react";
import { Sidebar, WORKSPACES } from "./Sidebar";
import type { Workspace } from "../../types";

const DESCRIPTIONS: Record<Workspace, string> = {
  analysis:
    "A transparent look at what happens between raw language and useful signals.",
  represent:
    "See how words become the numbers that classical models can work with.",
  classify:
    "Train a baseline classifier and inspect how it behaves on held-out examples.",
  embeddings:
    "Explore the geometry learned from word co-occurrence in your corpus.",
  language:
    "Measure how well a statistical language model predicts unseen text.",
};

export function AppShell({
  active,
  apiConnected,
  mobileNavOpen,
  onToggleNavigation,
  onChooseWorkspace,
  children,
}: {
  active: Workspace;
  apiConnected: boolean | null;
  mobileNavOpen: boolean;
  onToggleNavigation: () => void;
  onChooseWorkspace: (workspace: Workspace) => void;
  children: ReactNode;
}) {
  const currentWorkspace = WORKSPACES.find((item) => item.id === active)!;

  return (
    <div className="app-shell">
      {mobileNavOpen && (
        <button
          className="mobile-overlay"
          aria-label="Close navigation"
          onClick={onToggleNavigation}
        />
      )}
      <aside className={`sidebar ${mobileNavOpen ? "mobile-open" : ""}`}>
        <Sidebar
          active={active}
          onChoose={onChooseWorkspace}
        />
      </aside>
      <main className="main-area">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={onToggleNavigation}
            aria-label={mobileNavOpen ? "Close navigation" : "Open navigation"}
          >
            <Menu size={19} />
          </button>
          <div className="breadcrumbs">
            <span>Workspace</span>
            <span className="crumb-slash">/</span>
            <strong>{currentWorkspace.shortLabel}</strong>
          </div>
          <div className="topbar-right">
            <div
              className={`api-status ${apiConnected === false ? "offline" : ""}`}
              role="status"
            >
              <span className="status-pulse" />
              {apiConnected === null
                ? "Checking API"
                : apiConnected
                  ? "Python API connected"
                  : "Python API offline"}
            </div>
            <button
              className="icon-button help-button"
              title="About this workspace"
              aria-label="About this workspace"
            >
              <Info size={17} />
            </button>
            <div className="profile-avatar top-avatar">N</div>
          </div>
        </header>
        <div className="content-wrap">
          <section className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-line" />
                CLASSICAL NLP · {currentWorkspace.shortLabel.toUpperCase()}
              </div>
              <h1>{currentWorkspace.label}</h1>
              <p>{DESCRIPTIONS[active]}</p>
            </div>
            <div className="heading-meta">
              <span className="meta-dot" /> Local experiment
              <span className="meta-separator">·</span> No data stored
            </div>
          </section>
          {children}
          <section className="bottom-strip">
            <div className="strip-left">
              <div className="strip-icon">
                <BookOpen size={16} />
              </div>
              <div>
                <strong>Built for understanding, not just output.</strong>
                <span>Every feature uses a documented classical NLP method.</span>
              </div>
            </div>
            <div className="strip-right">
              <span className="stack-label">METHODS IN THIS WORKSPACE</span>
              <div className="method-pills">
                <span>NLTK</span>
                <span>scikit-learn</span>
                <span>Gensim</span>
              </div>
            </div>
          </section>
          <footer className="footer">
            <span>
              ATLAS <span className="footer-separator">/</span> CLASSICAL NLP
            </span>
            <span>
              Classical NLP workbench{" "}
              <span className="footer-separator">·</span> Local experiments
            </span>
          </footer>
        </div>
      </main>
    </div>
  );
}
