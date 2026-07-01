"""
SmartArch — services/chat_service.py

This file connects everything for the CHAT feature:
  1. Take the client's question
  2. Search ChromaDB for the most relevant room data  (rag_service)
  3. Send that data + the question to Gemini for a natural answer (ai_provider_service)
  4. Save the question + answer to the database (for chat history)
"""
from dao.FloorPlan_dao import FloorPlanDAO
from services.RAG_service import search_floor_plan_data
from services.AI_provider_service import generate_chat_answer
from config import Config


class ChatService:

    @staticmethod
    def answer_question(project_id: str, question: str) -> tuple:
        """
        Main entry point — called by the chat controller.
        Returns (response_dict, http_status_code)
        """
        question = (question or "").strip()
        if not question:
            return {"success": False, "message": "question is required."}, 400

        # ── Step 1: Make sure this floor plan exists and is ready ──
        floor_plan = FloorPlanDAO.get_by_id(project_id)
        if not floor_plan:
            return {"success": False, "message": "Floor plan not found."}, 404

        if floor_plan.status != "ready":
            return {
                "success": False,
                "message": f"This floor plan is still being analyzed (status: {floor_plan.status})."
            }, 202

        print(f"\n[CHAT] Project: {project_id} | Question: '{question}'")

        # ── Step 2: Search ChromaDB for relevant room data ──────
        context = search_floor_plan_data(project_id, question, top_k=3)

        if not context:
            # No stored data found at all — maybe RAG storage failed during upload
            answer = (
                "I don't have detailed information stored for this floor plan yet. "
                "Please contact your architect."
            )
        else:
            # ── Step 3: Generate the actual answer ──────────────
            answer = generate_chat_answer(
                question=question,
                context_text=context,
                project_name=floor_plan.project_name,
            )

        print(f"[CHAT] Answer: {answer}\n")

        # ── Step 4: Save this Q&A to the database ───────────────
        FloorPlanDAO.save_chat_message(
            project_id=project_id,
            query=question,
            answer=answer,
            language="en",
            model_used=Config.AI_PROVIDER,
        )

        return {
            "success": True,
            "data": {
                "project_id": project_id,
                "question": question,
                "answer": answer,
            }
        }, 200

    @staticmethod
    def get_history(project_id: str) -> tuple:
        floor_plan = FloorPlanDAO.get_by_id(project_id)
        if not floor_plan:
            return {"success": False, "message": "Floor plan not found."}, 404

        messages = FloorPlanDAO.get_chat_history(project_id)
        return {
            "success": True,
            "data": [m.to_dict() for m in messages]
        }, 200