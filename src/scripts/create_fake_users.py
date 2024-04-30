import argparse

import sys
sys.path.append('..')

from scripts import Session
from db.models import User
import hashlib

def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

parser = argparse.ArgumentParser(
    description='Create test users in database')
parser.add_argument(
    'n_users', metavar='nu', type=int, nargs="?", default=10, const=check_positive,
    help='Number of users to create')
parser.add_argument(
    'n_admins', metavar='na', type=int, nargs="?", default=5, const=check_positive,
    help='Number of admins to create')

args = parser.parse_args()

def create_users():
    with Session() as session:
        for i in range(args.n_users + args.n_admins):
            role = "user" if i < args.n_users else "admin"
            login = f'{role}_{i}'
            hashed_password = hashlib.sha256(login.encode()).hexdigest()

            user = User(
                username=login,
                password_hash=hashed_password,
                role=role
            )
            session.add(user)
            session.commit()
            print(f'{role.capitalize()} {user.username} added to database')
    
    print(f'Successfully added {args.n_users} users and {args.n_admins} admins')

if __name__ == "__main__":
    create_users()
