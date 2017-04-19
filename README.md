# Automatic options
An options-framework specifically designed for continuous control of walking simulations

Options are the RL word for re-useable sub-policies designed to acheive a specific goal. For example, in the path from your bed to work, you need to put on shoes, eat breakfast, turn on your car, etc. These are of great importance to learning complex tasks, but also for making lifelong learners, because options can be re-learned. 

This is a stab at a new framework for deep RL options. In essence, it divides learning a task into two subtasks: learning control, and learning which states to aim for. To learn control, the agent takes in a current state and a goal state, and learns a policy to navigate from one to the other. By learning this well across a large sample of current and goal states, the agent develops the ability to control its own motion.

Then, by learning which goal states to specify, the agent can learn to use this controller to maximize the task's reward. For example, the agent could learn to pick up one leg, place it down ahead of itself, pick up the other, and repeat. This is a form of options! In many cases, it is easier and more robust to learn the proper sequence of body positions than it is to learn the proper state-action mapping.

The real beauty of this framework is that the controller can be re-used. For example, the first task may be walking forward, and the second task may be walking backwards, but the state controller used in both will be the same (you just need to train a new goal-controller). Training on many tasks will actually improve the goal-controller for all of them! This is a form of transfer learning with positive transfer, which is rare in deep learning literature.

Much of the code is written, but this is still a work in progress. A brief outline below:

* **actor.py, critic.py, and actor_critic.py** contain the code for a generic actor-critic framework.
* **mujoco_actor_critic.py** teaches a humanoid to stand still. It works decently well, and by running it you can see the progress.
* **state_controller.py and goal_controller.py** manage the two tasks I mentioned above. **state_controller.py** manages controlling state, and **goal_controller.py** manages controlling the goal which state takes in.
* **movement_orchestrator.py** is the equivalent of actor_critic for this framework. It has methods for teaching the state_controller, teaching the goal_controller, AND for outputting goals and actions.


That's all for now. Stay tuned for more updates and results as they come out.
