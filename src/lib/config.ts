import { homedir } from 'os';
import { join } from 'path';

export interface KGConfig {
  vaultPath: string;
  dataDir: string;
  dbPath: string;
}

export interface ConfigOverrides {
  vaultPath?: string;
  dataDir?: string;
}

export function resolveConfig(overrides: ConfigOverrides): KGConfig {
  const vaultPath = overrides.vaultPath
    ?? process.env.KG_VAULT_PATH;

  if (!vaultPath) {
    throw new Error(
      'Vault path not configured. Set KG_VAULT_PATH or pass --vault-path.'
    );
  }

  const xdgData = process.env.XDG_DATA_HOME
    ?? join(homedir(), '.local', 'share');

  const dataDir = overrides.dataDir
    ?? process.env.KG_DATA_DIR
    ?? join(xdgData, 'knowledge-graph');

  return {
    vaultPath,
    dataDir,
    dbPath: join(dataDir, 'kg.db'),
  };
}
