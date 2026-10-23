import {
  Activity,
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  CircleHelp,
  FlaskConical,
  Layers3,
  Network,
  ScanText,
} from "lucide-react";
import type { Workspace, WorkspaceItem } from "../../types";

export const WORKSPACES: WorkspaceItem[] = [
  {
    id: "analysis",
    label: "Text analysis",
    shortLabel: "Analysis",
    icon: ScanText,
    group: "Explore",
  },
  {
    id: "represent",
    label: "Representations",
    shortLabel: "Representations",
    icon: Layers3,
    group: "Explore",
  },
  {
    id: "embeddings",
    label: "Word embeddings",
    shortLabel: "Embeddings",
    icon: Network,
    group: "Explore",
  },
  {
    id: "classify",
    label: "Text classification",
    shortLabel: "Classification",
    icon: Activity,
    group: "Evaluate",
  },
  {
    id: "language",
    label: "Language model",
    shortLabel: "Language model",
    icon: FlaskConical,
    group: "Evaluate",
  },
];

export function Sidebar({
  active,
  onChoose,
}: {
  active: Workspace;
  onChoose: (workspace: Workspace) => void;
}) {
  return (
    <>
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <div className="brand-name">ATLAS</div>
          <div className="brand-caption">NLP WORKBENCH</div>
        </div>
      </div>
      <div className="workspace-switcher">
        <div className="workspace-avatar">AT</div>
        <div className="workspace-name">
          <strong>ATLAS NLP</strong>
          <span>Personal workspace</span>
        </div>
        <ChevronDown size={15} />
      </div>
      {(["Explore", "Evaluate"] as const).map((group) => (
        <nav className="nav-group" key={group} aria-label={group}>
          <div className="nav-label">{group}</div>
          {WORKSPACES.filter((item) => item.group === group).map((item) => {
            const Icon = item.icon;
            return (
              <button
                aria-current={active === item.id ? "page" : undefined}
                className={`nav-item ${active === item.id ? "active" : ""}`}
                key={item.id}
                onClick={() => onChoose(item.id)}
              >
                <Icon size={17} strokeWidth={1.8} />
                <span>{item.label}</span>
                {active === item.id && (
                  <span className="nav-active-mark" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </nav>
      ))}
      <div className="sidebar-bottom">
        <div className="learning-note">
          <div className="learning-icon">
            <BookOpen size={16} />
          </div>
          <div>
            <strong>Classical NLP, clearly.</strong>
            <p>Learn the mechanics. Inspect every step.</p>
          </div>
        </div>
        <a
          className="sidebar-help"
          href="https://www.nltk.org/"
          target="_blank"
          rel="noreferrer"
        >
          <CircleHelp size={16} /> Documentation <ArrowUpRight size={13} />
        </a>
        <div className="profile-row">
          <div className="profile-avatar">N</div>
          <div className="profile-copy">
            <strong>Research space</strong>
            <span>Local workspace</span>
          </div>
          <span className="local-dot" title="Local API" />
        </div>
      </div>
    </>
  );
}
