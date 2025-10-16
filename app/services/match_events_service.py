"""
Servicio para consultar eventos del partido desde API externa
"""
import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import asyncio

# Esta clase simplemente representa un evento dentro del partido.
class MatchEvent:
    """Representa un evento del partido"""
    def __init__(self, event_type, minute, team, player, timestamp):
        self.event_type = event_type  # "goal", "corner", "foul", etc.
        self.minute = minute
        self.team = team
        self.player = player
        self.timestamp = timestamp
        
class FrameEvent:
    def __init__(self, frame_time, goal_event):
        self.frame_time = frame_time
        self.goal_event = goal_event

class MatchEventsService:
    """
    Servicio para conectar con API externa de eventos del partido
    """
    # Recibimos la direcccion de la apy y su llave
    def __init__(self, api_url: str, api_key: Optional[str] = None):
        self.api_url = api_url
        self.api_key = api_key
        self.cache_ttl = 5  # segundos - eventos recientes
        self._cache: Dict[str, tuple[datetime, List[MatchEvent]]] = {}
           
        
    def _parse_timestamp(self, ts_str: Optional[str]) -> datetime:
        """Parsea timestamp de la API"""
        if not ts_str: # In case our date string is none
            return datetime.now()
        try:
            # We have default dates in iso format, but most of the APIS return
            # the dates in format that includes the "Z" thats why we replace it 
            # with "+00:00", if it fails then we return now.
            return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except:
            return datetime.now()
        
        
    def _parse_events(self, data: Dict) -> List[MatchEvent]:
        """
        Parsea la respuesta de la API externa a objetos MatchEvent
        Adaptar según el formato de tu API
        """
        events = []
        for item in data:
            # Creamos una lista de events y vamos creando objetos MatchEvent a los cuales les asignamos 
            # los valores que vengan dentro de nuestra data, los adaptaremos para que funcionen con lo que nuestra API devuelva.
            
            # ***************************************** ¡Pendiente de adaptar!
            events.append(FrameEvent(
                frame_time=self.frameTime,
                goal_event=item.goalEvent
            ))
            
        return events
    
    
    async def get_recent_events(self, date_time_str: str) -> List[MatchEvent]:
        """
        Obtiene eventos recientes del partido (últimos N minutos)
        
        Args:
            match_id: ID del partido
            last_minutes: Ventana temporal para buscar eventos
        """
        
        middle_time = datetime.datetime.strptime(date_time_str, "%H:%M:%S.%f").time()
        
        start, end = middle_time - datetime.timedelta(seconds=0.5), middle_time + datetime.timedelta(seconds = 0.5)
        # Consultar API externa
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://localhost:8080/events/by-interval",
                    params={"start": start,
                            "end": end},
                )
                response.raise_for_status()
                data = response.json()
                
                events = self._parse_events(data) # hacemos el parsing de nuestros eventos y los guardamos en una lista
                return events
                
        except Exception as e:
            print(f"Error consultando eventos del partido: {e}")
            return []
    
    
    async def get_current_match_state(self, match_id: str) -> Dict:
        """
        Obtiene el estado actual del partido (minuto, marcador, etc.)
        """
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                response = await client.get(
                    f"{self.api_url}/matches/{match_id}",
                    headers=headers,
                    timeout=5.0
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error obteniendo estado del partido: {e}")
            return {"error": str(e)}


class MatchValidator:
    """
    Valida detecciones del modelo contra eventos reales del partido
    """
    def __init__(self, match_events_service: MatchEventsService):
        self.events_service = match_events_service
        
    async def validate_goal_detection(self, match_id: str, goal_minute: int, detected_team: Optional[str] = None) -> Dict:
        """
        Valida si un gol detectado por el modelo corresponde a un evento real
        
        Returns:
            {
                "is_valid": bool,
                "is_live": bool,
                "match_minute": int,
                "recent_goals": List[Dict],
                "confidence": float
            }
        """
        # Obtener eventos recientes (últimos 2 minutos)
        recent_events = await self.events_service.get_recent_events(match_id, last_minutes=2)
        match_state = await self.events_service.get_current_match_state(match_id)
        
        # Filtrar solo goles de los eventos que obtuvimos
        recent_goals = [e for e in recent_events if e.event_type == "goal"]
        
        # Validar si hay goles recientes, verificamos la cantidad de goles
        is_valid = len(recent_goals) > 0
        
        # Verificar si el partido está en vivo, ver si el partido es en vivo
        is_live = match_state.get("status") == "live"
        
        # Calcular confianza basada en timing, se asigna una confianza dependiendo del momento de ocurrencia del gol
        confidence = 1.0 if is_valid else 0.0
        if is_valid and recent_goals:
            # Mayor confianza si el gol fue muy reciente (< 30 segundos)
            latest_goal = recent_goals[0]
            seconds_ago = (datetime.now() - latest_goal.timestamp).seconds
            if seconds_ago < 30:
                confidence = 1.0
            elif seconds_ago < 60:
                confidence = 0.8
            else:
                confidence = 0.6
        
        # Validar equipo si se proporciona, para que no haya confusión entre los anotadores del gol
        team_match = False
        if detected_team and recent_goals:
            team_match = any(g.team.lower() == detected_team.lower() for g in recent_goals)
        
        return {
            "is_valid": is_valid,
            "is_live": is_live,
            "is_replay": is_live and not is_valid,  # Está en vivo pero no hay gol reciente
            "match_minute": match_state.get("minute", 0),
            "recent_goals": [
                {
                    "minute": g.minute,
                    "team": g.team,
                    "player": g.player,
                    "seconds_ago": (datetime.now() - g.timestamp).seconds
                }
                for g in recent_goals
            ],
            "confidence": confidence,
            "team_match": team_match if detected_team else None
        }
    
    async def validate_event_detection(self, match_id: str, event_type: str, detected_team: Optional[str] = None) -> Dict:
        """
        Valida cualquier tipo de evento detectado (gol, falta, córner, etc.)
        """
        
        # Obtenemos todos los eventos en los últimos 2 minutos
        recent_events = await self.events_service.get_recent_events(match_id, last_minutes=2)
        
        # verificamos los estados del partido
        match_state = await self.events_service.get_current_match_state(match_id)
        
        # Filtrar eventos del tipo específico
        matching_events = [e for e in recent_events if e.event_type == event_type]
        
        # Detectamos si hay el tipo de evento que buscamos
        is_valid = len(matching_events) > 0
        
        # Vemos si el partido es en vivo, es decir es una repetición o no.
        is_live = match_state.get("status") == "live"
        
        return {
            "is_valid": is_valid,
            "is_live": is_live,
            "is_replay": is_live and not is_valid,
            "match_minute": match_state.get("minute", 0),
            "matching_events": [
                {
                    "minute": e.minute,
                    "team": e.team,
                    "player": e.player
                }
                for e in matching_events
            ]
        }
    
        
    async def validate_detected_goal(self, mock_event: MatchEvent):
        
        """debo hacer la request a mi api externa"""
        
        recent_events = await self.events_service.get_recent_events(mock_event.timestamp)
        
        c = 0
        f = 0
        for e in recent_events:
            if e.goal_event:
                c += 1
            else:
                f += 1
        
        goal_classif = False
        if (c > f):
            goal_classif = True
            
                
        return {
            "live_goal": goal_classif,
        }
            