from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
    name : str
    age : int

users = []


@app.post("/users")
def create_user(user: User):
    users.append(user)
    return {"message": f"user {user.name} created"}

# Rota GET para listar ou dar as boas-vindas (nome alterado para get_users)
@app.get("/users")
def get_users():
    return users
