import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
import time

# --- Constants ---
WIDTH, HEIGHT = 1000, 700
BG_COLOR = "#2c3e50"
CANVAS_COLOR = "#34495e"
TEXT_COLOR = "#ecf0f1"
POINT_COLOR = "#e74c3c"
DEPOT_COLOR = "#f1c40f"
DRONE_COLOR = "#3498db"
ROUTE_COLOR = "#2ecc71"
FONT_NORMAL = ("Helvetica", 10)
FONT_BOLD = ("Helvetica", 12, "bold")

# --- Genetic Algorithm Parameters ---
POPULATION_SIZE = 50
ELITISM_RATE = 0.1
MUTATION_RATE = 0.02
GENERATIONS = 100

# --- Drone Parameters ---
DRONE_SPEED = 50  # pixels per second
MAX_BATTERY = 1000 # arbitrary units
BATTERY_CONSUMPTION_RATE = 1 # units per second
WEATHER_CONDITIONS = {"Clear": 1.0, "Windy": 1.2, "Rainy": 1.5}


class City:
    """Represents a delivery point in the city."""
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = name

    def distance_to(self, city):
        """Calculates Euclidean distance to another city."""
        return math.sqrt((self.x - city.x)**2 + (self.y - city.y)**2)


class Route:
    """Represents a delivery route (a chromosome in GA)."""
    def __init__(self, cities):
        self.cities = cities
        self.distance = 0
        self.fitness = 0.0
        self.calculate_metrics()

    def calculate_metrics(self):
        """Calculates the total distance and fitness of the route."""
        if not self.cities:
            return
        # Calculate total distance
        path_distance = 0
        for i in range(len(self.cities)):
            from_city = self.cities[i]
            to_city = self.cities[i + 1] if i + 1 < len(self.cities) else self.cities[0]
            path_distance += from_city.distance_to(to_city)
        self.distance = path_distance

        # Calculate fitness (inverse of distance)
        self.fitness = 1 / float(self.distance) if self.distance > 0 else 0


class GeneticAlgorithm:
    """Handles the route optimization using a genetic algorithm."""

    def create_initial_population(self, depot, delivery_points):
        """Creates the initial population of random routes."""
        population = []
        base_cities = delivery_points[:]
        for _ in range(POPULATION_SIZE):
            random.shuffle(base_cities)
            # Route always starts and ends at the depot
            route_cities = [depot] + base_cities
            population.append(Route(route_cities))
        return population

    def rank_routes(self, population):
        """Ranks routes by fitness."""
        return sorted(population, key=lambda x: x.fitness, reverse=True)

    def selection(self, ranked_population):
        """Selects parents for the next generation using elitism and roulette wheel."""
        selection_results = []
        # Elitism: carry over the best routes
        elite_size = int(len(ranked_population) * ELITISM_RATE)
        for i in range(elite_size):
            selection_results.append(ranked_population[i])

        # Roulette Wheel Selection for the rest
        fitness_sum = sum(route.fitness for route in ranked_population)
        for _ in range(len(ranked_population) - elite_size):
            pick = random.uniform(0, fitness_sum)
            current = 0
            for route in ranked_population:
                current += route.fitness
                if current > pick:
                    selection_results.append(route)
                    break
        return selection_results

    def crossover(self, parent1, parent2):
        """Creates a child route from two parents using ordered crossover."""
        child_p1 = []
        
        # Ensure depot is always at the start
        parent1_genes = parent1.cities[1:]
        parent2_genes = parent2.cities[1:]
        
        gene_a = int(random.random() * len(parent1_genes))
        gene_b = int(random.random() * len(parent1_genes))
        
        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)

        for i in range(start_gene, end_gene):
            child_p1.append(parent1_genes[i])
            
        child_p2 = [item for item in parent2_genes if item not in child_p1]

        child_cities = [parent1.cities[0]] + child_p1 + child_p2
        return Route(child_cities)

    def breed_population(self, mating_pool):
        """Breeds a new population from the mating pool."""
        children = []
        pool = random.sample(mating_pool, len(mating_pool))
        elite_size = int(len(mating_pool) * ELITISM_RATE)
        
        # Elitism
        for i in range(elite_size):
            children.append(mating_pool[i])
        
        # Crossover
        for i in range(len(mating_pool) - elite_size):
            child = self.crossover(pool[i], pool[len(mating_pool)-i-1])
            children.append(child)
        return children

    def mutate(self, individual, mutation_rate):
        """Mutates a route by swapping two cities (excluding the depot)."""
        if random.random() < mutation_rate:
            genes = individual.cities[1:] # Exclude depot
            if len(genes) > 1:
                swap_with = int(random.random() * len(genes))
                swap_from = int(random.random() * len(genes))
                genes[swap_from], genes[swap_with] = genes[swap_with], genes[swap_from]
            individual.cities = [individual.cities[0]] + genes
            individual.calculate_metrics()
        return individual

    def mutate_population(self, population):
        """Applies mutation to the entire population."""
        mutated_pop = []
        for ind in range(len(population)):
            mutated_ind = self.mutate(population[ind], MUTATION_RATE)
            mutated_pop.append(mutated_ind)
        return mutated_pop

    def next_generation(self, current_gen):
        """Creates the next generation of routes."""
        ranked_pop = self.rank_routes(current_gen)
        selected_pop = self.selection(ranked_pop)
        mating_pool = self.breed_population(selected_pop)
        next_gen = self.mutate_population(mating_pool)
        return next_gen


class Drone:
    """Represents the delivery drone."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.battery = MAX_BATTERY
        self.route = []
        self.target_city_index = 0
        self.is_moving = False

    def move_to_target(self, target_x, target_y, speed_factor):
        """Moves the drone towards a target coordinate."""
        if not self.is_moving:
            return True # Reached target

        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        
        effective_speed = DRONE_SPEED / speed_factor
        
        if dist < effective_speed / 10: # Close enough
            self.x = target_x
            self.y = target_y
            return True
        else:
            self.x += (dx / dist) * (effective_speed / 10)
            self.y += (dy / dist) * (effective_speed / 10)
            self.consume_battery(speed_factor)
            return False

    def consume_battery(self, weather_factor):
        """Reduces battery based on consumption rate and weather."""
        self.battery -= BATTERY_CONSUMPTION_RATE * weather_factor
        if self.battery < 0:
            self.battery = 0


class DroneSimulator(tk.Tk):
    """Main application class for the simulator."""
    def __init__(self):
        super().__init__()
        self.title("Drone Delivery Route Optimization Simulator")
        self.geometry(f"{WIDTH}x{HEIGHT}")
        self.configure(bg=BG_COLOR)

        self.depot = None
        self.delivery_points = []
        self.optimized_route = None
        self.drone = None
        self.ga = GeneticAlgorithm()
        self.is_simulating = False
        self.current_weather = "Clear"
        
        self.create_widgets()
        self.canvas.bind("<Button-1>", self.add_point)

    def create_widgets(self):
        """Creates all GUI widgets."""
        # --- Main Layout ---
        main_frame = tk.Frame(self, bg=BG_COLOR)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(main_frame, bg=CANVAS_COLOR, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        dashboard_frame = tk.Frame(main_frame, width=250, bg=BG_COLOR)
        dashboard_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        dashboard_frame.pack_propagate(False)

        # --- Dashboard Widgets ---
        title_label = tk.Label(dashboard_frame, text="Dashboard", font=("Helvetica", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
        title_label.pack(pady=10)

        # Controls
        controls_frame = ttk.LabelFrame(dashboard_frame, text="Controls")
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.start_btn = ttk.Button(controls_frame, text="Optimize & Simulate", command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, padx=5, pady=5)
        
        self.reset_btn = ttk.Button(controls_frame, text="Reset", command=self.reset)
        self.reset_btn.pack(fill=tk.X, padx=5, pady=5)
        
        # Info
        info_frame = ttk.LabelFrame(dashboard_frame, text="Simulation Info")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.distance_label = tk.Label(info_frame, text="Total Distance: N/A", anchor="w", font=FONT_NORMAL)
        self.distance_label.pack(fill=tk.X)
        self.time_label = tk.Label(info_frame, text="Est. Time: N/A", anchor="w", font=FONT_NORMAL)
        self.time_label.pack(fill=tk.X)
        self.generation_label = tk.Label(info_frame, text="Generation: 0", anchor="w", font=FONT_NORMAL)
        self.generation_label.pack(fill=tk.X)

        # Drone Status
        drone_frame = ttk.LabelFrame(dashboard_frame, text="Drone Status")
        drone_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.battery_label = tk.Label(drone_frame, text=f"Battery: {MAX_BATTERY}", anchor="w", font=FONT_NORMAL)
        self.battery_label.pack(fill=tk.X)
        self.battery_progress = ttk.Progressbar(drone_frame, orient='horizontal', length=200, mode='determinate', maximum=MAX_BATTERY)
        self.battery_progress['value'] = MAX_BATTERY
        self.battery_progress.pack(fill=tk.X, pady=5)

        # Weather
        weather_frame = ttk.LabelFrame(dashboard_frame, text="Environment")
        weather_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(weather_frame, text="Weather:", anchor="w").pack(fill=tk.X)
        self.weather_var = tk.StringVar(value=self.current_weather)
        weather_menu = ttk.OptionMenu(weather_frame, self.weather_var, self.current_weather, *WEATHER_CONDITIONS.keys(), command=self.change_weather)
        weather_menu.pack(fill=tk.X, padx=5, pady=5)

    def add_point(self, event):
        """Adds a delivery point or depot on canvas click."""
        if self.is_simulating:
            return
            
        x, y = event.x, event.y
        if not self.depot:
            self.depot = City(x, y, "Depot")
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=DEPOT_COLOR, outline=DEPOT_COLOR, tags="point")
            self.canvas.create_text(x, y-10, text="Depot", fill=TEXT_COLOR, font=FONT_NORMAL)
        else:
            name = f"P{len(self.delivery_points)+1}"
            self.delivery_points.append(City(x, y, name))
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill=POINT_COLOR, outline=POINT_COLOR, tags="point")
            self.canvas.create_text(x, y-10, text=name, fill=TEXT_COLOR, font=FONT_NORMAL)

    def draw_route(self, route):
        """Draws a route on the canvas."""
        self.canvas.delete("route")
        if not route or not route.cities:
            return
        
        city_coords = [(c.x, c.y) for c in route.cities]
        # Add depot at the end to close the loop for visualization
        city_coords.append((self.depot.x, self.depot.y))
        
        self.canvas.create_line(city_coords, fill=ROUTE_COLOR, width=2, tags="route")

    def start_simulation(self):
        """Starts the optimization and simulation process."""
        if len(self.delivery_points) < 2:
            messagebox.showwarning("Warning", "Please add at least 2 delivery points.")
            return

        if self.is_simulating:
            return
            
        self.is_simulating = True
        self.start_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.DISABLED)
        
        self.drone = Drone(self.depot.x, self.depot.y)
        self.canvas.create_oval(self.drone.x-5, self.drone.y-5, self.drone.x+5, self.drone.y+5, fill=DRONE_COLOR, tags="drone")
        
        self.run_ga()

    def run_ga(self, resume_from_drone=False):
        """Executes the genetic algorithm to find the optimal route."""
        start_time = time.time()
        
        if resume_from_drone:
            # Re-optimizing mid-route
            drone_location_city = City(self.drone.x, self.drone.y, "Drone")
            remaining_points = self.optimized_route.cities[self.drone.target_city_index:]
            if not remaining_points: # If last point was the target
                 self.end_simulation()
                 return
            pop = self.ga.create_initial_population(drone_location_city, remaining_points)
        else:
            # Initial optimization from depot
            pop = self.ga.create_initial_population(self.depot, self.delivery_points)

        initial_distance = self.ga.rank_routes(pop)[0].distance
        
        for i in range(GENERATIONS):
            pop = self.ga.next_generation(pop)
            best_route = self.ga.rank_routes(pop)[0]
            self.draw_route(best_route)
            self.generation_label.config(text=f"Generation: {i+1}/{GENERATIONS}")
            self.distance_label.config(text=f"Best Distance: {best_route.distance:.2f}")
            self.update_idletasks() # Force UI update

        self.optimized_route = self.ga.rank_routes(pop)[0]
        
        if resume_from_drone:
             # Splice the new sub-route into the main route
             # The first city in the new optimized route is the drone's current location, so we skip it.
             new_path = self.optimized_route.cities[1:]
             # The old path up to the current target
             old_path_prefix = self.optimized_route.cities[:self.drone.target_city_index]
             # Combine them
             self.optimized_route.cities = old_path_prefix + new_path
             self.optimized_route.calculate_metrics()
             # The drone continues to its original next target, but the path *after* that is now re-optimized.
        
        self.distance_label.config(text=f"Total Distance: {self.optimized_route.distance:.2f}")
        est_time = self.optimized_route.distance / DRONE_SPEED
        self.time_label.config(text=f"Est. Time: {est_time:.2f}s")
        
        print(f"--- GA Finished in {time.time() - start_time:.2f}s ---")
        print(f"Initial distance: {initial_distance:.2f}")
        print(f"Optimized distance: {self.optimized_route.distance:.2f}")

        # Start drone movement simulation
        if not resume_from_drone:
            self.drone.route = self.optimized_route.cities + [self.depot]
            self.drone.target_city_index = 1 # Start with the first delivery point
            self.drone.is_moving = True
        
        self.animate_drone()


    def animate_drone(self):
        """Animates the drone's movement along the route."""
        if not self.drone or not self.drone.is_moving:
            return

        if self.drone.target_city_index >= len(self.drone.route):
            self.end_simulation()
            return
            
        target_city = self.drone.route[self.drone.target_city_index]
        
        # Check for re-routing conditions
        if self.drone.battery < MAX_BATTERY * 0.2:
            messagebox.showinfo("Low Battery", "Low battery! Re-routing back to depot.")
            self.reroute_to_depot()
            return

        weather_factor = WEATHER_CONDITIONS[self.current_weather]
        reached_target = self.drone.move_to_target(target_city.x, target_city.y, weather_factor)
        
        self.canvas.coords("drone", self.drone.x-5, self.drone.y-5, self.drone.x+5, self.drone.y+5)
        
        # Update dashboard
        self.battery_label.config(text=f"Battery: {self.drone.battery:.0f}/{MAX_BATTERY}")
        self.battery_progress['value'] = self.drone.battery

        if self.drone.battery <= 0:
            messagebox.showerror("Simulation End", "Drone out of battery!")
            self.end_simulation()
            return

        if reached_target:
            print(f"Reached {target_city.name}")
            self.drone.target_city_index += 1
            
        self.after(50, self.animate_drone)

    def end_simulation(self):
        self.drone.is_moving = False
        self.is_simulating = False
        self.reset_btn.config(state=tk.NORMAL)
        print("--- Simulation Finished ---")
        if self.drone.target_city_index >= len(self.drone.route):
             messagebox.showinfo("Success", "All deliveries completed and returned to depot!")

    def reroute_to_depot(self):
        """Forces the drone to re-route directly to the depot."""
        self.drone.route = self.drone.route[:self.drone.target_city_index] + [self.depot]
        # self.drone.target_city_index remains the same, it will now point to the newly added depot
        self.optimized_route.cities = self.drone.route
        self.optimized_route.calculate_metrics()
        self.draw_route(self.optimized_route)
        self.distance_label.config(text=f"Total Distance: {self.optimized_route.distance:.2f}")
        self.animate_drone()

    def change_weather(self, selected_weather):
        """Handles weather changes and potential re-routing."""
        self.current_weather = selected_weather
        print(f"Weather changed to: {self.current_weather}")
        if self.is_simulating and self.drone.is_moving:
            if self.current_weather == "Rainy":
                messagebox.showwarning("Weather Alert", "Heavy rain detected! Re-optimizing route for safety and efficiency.")
                # Stop current movement to re-calculate
                self.drone.is_moving = False
                self.run_ga(resume_from_drone=True)
                self.drone.is_moving = True
                self.animate_drone()

    def reset(self):
        """Resets the entire simulation."""
        self.is_simulating = False
        if self.drone:
            self.drone.is_moving = False
        self.canvas.delete("all")
        self.depot = None
        self.delivery_points = []
        self.optimized_route = None
        self.drone = None
        self.distance_label.config(text="Total Distance: N/A")
        self.time_label.config(text="Est. Time: N/A")
        self.generation_label.config(text="Generation: 0")
        self.battery_label.config(text=f"Battery: {MAX_BATTERY}")
        self.battery_progress['value'] = MAX_BATTERY
        self.start_btn.config(state=tk.NORMAL)
        print("--- Simulation Reset ---")


if __name__ == "__main__":
    app = DroneSimulator()
    app.mainloop()
