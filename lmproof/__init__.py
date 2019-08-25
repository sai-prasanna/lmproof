from .proof_reader import ProofReader

def load(language: str, gpu: bool = False) -> ProofReader:
    return ProofReader.load(language, gpu)
