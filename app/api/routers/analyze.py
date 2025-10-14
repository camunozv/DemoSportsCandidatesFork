# ApiRouter: Allows endpoint grouping 
# UploadFile: Is an special module for file manip. Allows accessing some asynchronous methods
# like read and file metadata like .content_type
# File: Dependency marker that tells a parameter comes from  multipart/form-data
from fastapi import APIRouter, UploadFile, File, HTTPException

# Allows decoding image bytes into an image object thus we can manipulate it
from PIL import Image

# Libraries for byte manipulation (io) for useful to Image to read files
# Time: precise time measuring library
import io, time

from app.services.analysis_service import AnalysisService
from app.schemas.io import CompleteResponse

# We create the router object and group all of them under the tag "analyze"
router = APIRouter(tags=["analyze"])

# Allows us to get the AnalysisServive object that we are supposed to implement.
def get_service() -> AnalysisService:
    # Podrías inyectar por deps y vida-útil global
    from app.api.deps import analysis_service
    return analysis_service()

# Defines the endpoints and the response type, which is a pydantic model.
@router.post("/analyze", response_model=CompleteResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(await file.read()))
        resp = get_service().analyze(img)
        print(f"analyze ms={(time.perf_counter()-t0)*1000:.2f}")
        return resp
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {e}")