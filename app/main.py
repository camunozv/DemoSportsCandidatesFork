from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Imports a function that returns the settings from the core/config.py
from app.core.config import get_settings

# Imports routers from the api/routers folder
from app.api.routers import analyze, health, ask, validate

def create_app() -> FastAPI:
    
    # Here we get the settings of the config.py file, that file contains sort of environment variables
    s = get_settings()
    app = FastAPI(title="Complete Soccer Analysis API", version="2.0.0")
    
    # Sets up from which adresses we can receive requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.CORS_ORIGINS, # configured origins in the settings file
        allow_credentials=True, # allow credentials on requests -> cookies, authorization headers
        allow_methods=["GET","POST","PUT","DELETE","OPTIONS"], # allowed http methods
        allow_headers=["*"], 
    )
    
    # Registers imported routers within the app
    # per router we have particular routes defined
    app.include_router(analyze.router)
    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(validate.router)  # NUEVO: Endpoints de validación

    @app.get("/")
    def root():
        return {
            "message": "Complete Soccer Analysis API",
            "version": "2.0.0",
            "endpoints": ["/analyze", "/health", "/ask", "/docs"],
        }
    
    # NUEVO: Pre-carga de modelos al iniciar la aplicación
    @app.on_event("startup")
    async def startup_event():
        print(" Iniciando Complete Soccer Analysis API v2.0...")
        print(" Pre-cargando modelos...")
        
        # Esto fuerza la carga de los modelos en el @lru_cache
        from app.api.deps import analysis_service
        service = analysis_service()
        
        print(f" Modelos cargados:")
        print(f"  - Reconocimiento facial: {service.face_rec.loaded}")
        print(f"  - Goal Classifier: {service.goal_clf.model is not None}")
        print(f"  - Jersey Detector (YOLO): {service.jersey_det.yolo is not None}")
        print(" Sistema listo!")
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)