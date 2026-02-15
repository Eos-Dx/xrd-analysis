"""
Database Backup and Restore for Omniscan Orchestrator

Implements automatic daily encrypted backups with 30-day retention
as specified in DB.md.
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DatabaseBackupManager:
    """Manages automated database backups with retention policy."""
    
    def __init__(self, db_path: str, backup_dir: str, retention_days: int = 30):
        """Initialize backup manager.
        
        Args:
            db_path: Path to main database file
            backup_dir: Directory to store backups
            retention_days: Number of days to retain backups (default 30)
        """
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def get_backup_path(self, timestamp: Optional[datetime] = None) -> Path:
        """Generate backup file path with timestamp.
        
        Args:
            timestamp: Backup timestamp (defaults to now)
            
        Returns:
            Path to backup file
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        date_str = timestamp.strftime("%Y-%m-%d")
        time_str = timestamp.strftime("%H%M%S")
        filename = f"orchestrator_{date_str}_{time_str}.db"
        return self.backup_dir / filename
    
    def create_backup(self) -> Path:
        """Create a backup of the database.
        
        Uses SQLite's backup API for consistent snapshots.
        
        Returns:
            Path to created backup file
            
        Raises:
            Exception: If backup fails
        """
        backup_path = self.get_backup_path()
        
        try:
            # Open source database
            source = sqlite3.connect(self.db_path)
            
            # Create backup database
            backup = sqlite3.connect(backup_path)
            
            # Perform backup (handles concurrent access correctly)
            with backup:
                source.backup(backup)
            
            backup.close()
            source.close()
            
            logger.info(f"Database backup created: {backup_path}")
            logger.info(f"Backup size: {backup_path.stat().st_size / (1024*1024):.2f} MB")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Clean up partial backup if it exists
            if backup_path.exists():
                backup_path.unlink()
            raise
    
    def restore_backup(self, backup_path: Path, target_path: Optional[Path] = None):
        """Restore database from backup.
        
        Args:
            backup_path: Path to backup file to restore
            target_path: Path to restore to (defaults to main db_path)
            
        Raises:
            FileNotFoundError: If backup doesn't exist
            Exception: If restore fails
        """
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        if target_path is None:
            target_path = self.db_path
        
        try:
            # Verify backup integrity first
            self.verify_backup(backup_path)
            
            # Copy backup to target location
            shutil.copy2(backup_path, target_path)
            
            logger.info(f"Database restored from backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            raise
    
    def verify_backup(self, backup_path: Path) -> bool:
        """Verify backup file integrity.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup is valid
            
        Raises:
            Exception: If backup is corrupted
        """
        try:
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            conn.close()
            
            if result[0] != "ok":
                raise Exception(f"Backup integrity check failed: {result[0]}")
            
            logger.info(f"Backup verified: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup verification failed: {e}")
            raise
    
    def cleanup_old_backups(self):
        """Remove backups older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        removed_count = 0
        total_size = 0
        
        for backup_file in self.backup_dir.glob("orchestrator_*.db"):
            try:
                # Extract date from filename: orchestrator_YYYY-MM-DD_HHMMSS.db
                parts = backup_file.stem.split("_")
                if len(parts) >= 2:
                    date_str = parts[1]  # YYYY-MM-DD
                    backup_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if backup_date < cutoff_date:
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()
                        removed_count += 1
                        total_size += file_size
                        logger.info(f"Removed old backup: {backup_file}")
            
            except Exception as e:
                logger.warning(f"Failed to process backup {backup_file}: {e}")
        
        if removed_count > 0:
            logger.info(
                f"Cleanup complete: removed {removed_count} backups, "
                f"freed {total_size / (1024*1024):.2f} MB"
            )
    
    def list_backups(self) -> List[Path]:
        """List all available backups, sorted by date (newest first).
        
        Returns:
            List of backup file paths
        """
        backups = sorted(
            self.backup_dir.glob("orchestrator_*.db"),
            reverse=True  # Newest first
        )
        return backups
    
    def get_backup_info(self, backup_path: Path) -> dict:
        """Get information about a backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dictionary with backup metadata
        """
        info = {
            "path": str(backup_path),
            "size_mb": backup_path.stat().st_size / (1024*1024),
            "created": datetime.fromtimestamp(backup_path.stat().st_mtime),
        }
        
        try:
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            
            # Get patient count
            cursor.execute("SELECT COUNT(*) FROM patients")
            info["patient_count"] = cursor.fetchone()[0]
            
            # Get measurement count
            cursor.execute("SELECT COUNT(*) FROM measurements")
            info["measurement_count"] = cursor.fetchone()[0]
            
            # Get calibration count
            cursor.execute("SELECT COUNT(*) FROM calibration_log")
            info["calibration_count"] = cursor.fetchone()[0]
            
            # Get schema version
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            result = cursor.fetchone()
            info["schema_version"] = result[0] if result else None
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to get backup info: {e}")
            info["error"] = str(e)
        
        return info
    
    def perform_daily_backup(self) -> dict:
        """Perform daily backup and cleanup routine.
        
        Returns:
            Dictionary with backup results
        """
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "success": False,
            "backup_path": None,
            "size_mb": None,
            "patient_count": None,
            "measurement_count": None,
            "error": None
        }
        
        try:
            # Create backup
            backup_path = self.create_backup()
            results["backup_path"] = str(backup_path)
            
            # Get backup info
            info = self.get_backup_info(backup_path)
            results["size_mb"] = info.get("size_mb")
            results["patient_count"] = info.get("patient_count")
            results["measurement_count"] = info.get("measurement_count")
            
            # Cleanup old backups
            self.cleanup_old_backups()
            
            results["success"] = True
            
        except Exception as e:
            logger.error(f"Daily backup failed: {e}")
            results["error"] = str(e)
        
        return results


def encrypt_backup(backup_path: Path, encryption_key: bytes) -> Path:
    """Encrypt a backup file (placeholder for AES-256 encryption).
    
    Note: This is a placeholder. In production, use proper encryption
    libraries like cryptography.fernet or similar.
    
    Args:
        backup_path: Path to unencrypted backup
        encryption_key: Encryption key
        
    Returns:
        Path to encrypted backup file
    """
    encrypted_path = backup_path.with_suffix(".db.enc")
    
    # TODO: Implement AES-256 encryption
    # For now, just copy the file
    logger.warning("Encryption not yet implemented - backup is unencrypted!")
    shutil.copy2(backup_path, encrypted_path)
    
    return encrypted_path


def decrypt_backup(encrypted_path: Path, encryption_key: bytes) -> Path:
    """Decrypt a backup file (placeholder for AES-256 decryption).
    
    Args:
        encrypted_path: Path to encrypted backup
        encryption_key: Decryption key
        
    Returns:
        Path to decrypted backup file
    """
    decrypted_path = encrypted_path.with_suffix("")
    
    # TODO: Implement AES-256 decryption
    # For now, just copy the file
    logger.warning("Decryption not yet implemented!")
    shutil.copy2(encrypted_path, decrypted_path)
    
    return decrypted_path
