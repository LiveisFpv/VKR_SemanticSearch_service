from src.lib.logger import Logger
from src.http.grpc import service_pb2, service_pb2_grpc
import grpc

class SemanticServiceHandlerGrpc(service_pb2_grpc.SemanticServiceServicer):
    def __init__(self, semantic_service, logger: Logger):
        self.logger = logger

    def SearchPaper(self, request:service_pb2.SearchRequest, context:grpc.ServicerContext)->service_pb2.PapersResponse:
        self.logger.info(f"SearchPaper request: {request.Input_data}")

        matching_papers = []
        # Отправляем данные из БД релевантные по сходству
        return service_pb2.PapersResponse(
            Papers=[service_pb2.PaperResponse(
                ID="W157958743",
                Title="paper.Title",
                Abstract="paper.Abstract",
                Year=2024,
                Best_oa_location="paper.Best_oa_location"
            )]
        )

    def AddPaper(self, request:service_pb2.AddRequest, context:grpc.ServicerContext)->service_pb2.ErrorResponse:
        self.logger.info(f"AddPaper request: {request.Title}")

        # Создаём объект статьи
        paper = service_pb2.PaperResponse(
            ID=request.ID,
            Title=request.Title,
            Abstract=request.Abstract,
            Year=request.Year,
            Best_oa_location=request.Best_oa_location
        )

        # Добавляем в БД и проводим векторизацию

        return service_pb2.ErrorResponse(
            Error=""  # Пустая строка = нет ошибки
        )
