## Communication Strategies in Multi-Agent Reinforcement Learning Systems
Members: Ming Liu, Emily Bejerano, Andrea Su, Pair Phongphaew

This repository includes the environmental and test code used to test multiple communication
strategies in the report (also included here).

The ma-gym environment (https://github.com/koulanurag/ma-gym) is used. Install that repo:

git clone https://github.com/koulanurag/ma-gym.git

cd ma-gym

pip install -e .

Replace the lumberjacks.py file in the library with the file in this 'env' folder. 
Our environment modifies the original lumberjacks to fully implement walls/obstacles.

Some code based off https://github.com/koulanurag/minimal-marl implementation
