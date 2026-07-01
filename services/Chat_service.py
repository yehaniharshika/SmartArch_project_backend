"""
SmartArch — services/Chat_service.py

Orchestrates the full chat flow:
  1. Validate the floor plan exists and is ready
  2. Search ChromaDB for relevant context (RAG_service)
  3. Generate answer using Gemini LLM (AI_provider_service)
  4. Save Q&A to database
  5. Return answer to controller
"""
from dao.FloorPlan_dao import FloorPlanDAO
from services.RAG_service import search_floor_plan_data
from services.AI_provider_service import generate_chat_answer
from config import Config


class ChatService:

    @staticmethod
    def answer_question(project_id: str, question: str) -> tuple:
        """
        Main entry point — called by Chat_controller.
        Returns (response_dict, http_status_code).
        """
        question = (question or "").strip()
        if not question:
            return {"success": False, "message": "question is required."}, 400

        # Step 1: Validate floor plan
        floor_plan = FloorPlanDAO.get_by_id(project_id)
        if not floor_plan:
            return {"success": False, "message": "Floor plan not found."}, 404

        if floor_plan.status != "ready":
            return {
                "success": False,
                "message": (
                    f"This floor plan is still being analyzed "
                    f"(status: {floor_plan.status}). Please try again shortly."
                ),
            }, 202

        print(f"\n[CHAT] Project: {project_id} | Q: '{question}'")

        # Step 2: Retrieve relevant context from ChromaDB
        context = search_floor_plan_data(project_id, question, top_k=4)

        # Step 3: Generate answer
        if not context:
            answer = (
                "I don't have detailed floor plan data stored for this project yet. "
                "Please contact your architect to re-upload the plan."
            )
        else:
            answer = generate_chat_answer(
                question=question,
                context_text=context,
                project_name=floor_plan.project_name,
            )

        print(f"[CHAT] Answer: {answer[:100]}...")

        # Step 4: Save to DB
        try:
            FloorPlanDAO.save_chat_message(
                project_id=project_id,
                query=question,
                answer=answer,
                language="en",
                model_used=Config.AI_PROVIDER,
            )
        except Exception as e:
            print(f"[CHAT] Save failed (non-critical): {e}")

        return {
            "success": True,
            "data": {
                "project_id": project_id,
                "question":   question,
                "answer":     answer,
            },
        }, 200

    @staticmethod
    def get_history(project_id: str) -> tuple:
        """Returns full chat history for a project."""
        floor_plan = FloorPlanDAO.get_by_id(project_id)
        if not floor_plan:
            return {"success": False, "message": "Floor plan not found."}, 404

        messages = FloorPlanDAO.get_chat_history(project_id)
        return {
            "success": True,
            "data": [m.to_dict() for m in messages],
        }, 200