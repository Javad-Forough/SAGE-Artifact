from __future__ import annotations


class SAGE_TPM:
    """
    TPM-backed SAGE wrapper.
    """

    def __init__(
        self,
        db_path: str,
        root_key_sealed: str = "sealed_root_key.sage_tpm.bin",
        epochs_path: str = "epochs_tpm.json",
        tpm_blob_prefix: str = "results/tpm",
    ):
        # Lazy imports to avoid circular / import-order issues.
        from service_tpm import SealedAgentMemoryServiceTPM, ServiceConfig
        from tpm_epoch import TPMEpochStore
        from tpm_sealer import TPMSealer

        self.epoch = TPMEpochStore(
            state_path=epochs_path,
            blob_prefix=tpm_blob_prefix,
        )
        self.sealer = TPMSealer(
            blob_prefix=tpm_blob_prefix,
        )

        self.mem = SealedAgentMemoryServiceTPM(
            ServiceConfig(
                root_key_sealed_path=root_key_sealed,
                db_path=db_path,
            ),
            epoch=self.epoch,
            sealer=self.sealer,
        )

    def put(self, scope_id: str, payload: dict, kind: str = "fact"):
        return self.mem.put(scope_id, payload, kind=kind)

    def put_derived(
        self,
        scope_id: str,
        payload: dict,
        kind: str,
        derived_from_item_ids: list[str],
        source_scope_ids: list[str] | None = None,
    ):
        return self.mem.put_derived(
            scope_id,
            payload,
            kind=kind,
            derived_from_item_ids=derived_from_item_ids,
            source_scope_ids=source_scope_ids,
        )

    def get_recent(self, scope_id: str, limit: int = 10):
        return self.mem.get_recent(scope_id, limit=limit)

    def forget_scope(
        self,
        scope_id: str,
        delete_ciphertext_rows: bool = True,
        propagate: bool = True,
    ):
        return self.mem.forget_scope(
            scope_id,
            delete_ciphertext_rows=delete_ciphertext_rows,
            propagate=propagate,
        )

    def close(self):
        if hasattr(self.mem.store, "close"):
            try:
                self.mem.store.close()
            except Exception:
                pass
