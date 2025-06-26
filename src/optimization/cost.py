class CostFunction:
    def __init__(self, parent):
        self.parent = parent

    def cost_fn(self):
        total_cost = 0
        for key, slot in self.parent.slots.items():
            day_type, hour, direction = key
            geo_mean_ons = self.parent.geo_lookup.get((day_type, hour, direction), 0)
            trains = slot.current_trains

            if trains < self.parent.min_trains or trains > slot.max_slots:
                total_cost += self.parent.penalties['constraint_violation']
                continue

            if trains > 0:
                load_per_train = geo_mean_ons / trains
                overload = max(0, load_per_train - self.parent.train_capacity)
                total_cost += overload**2 * self.parent.penalties['overload']

                underutil = max(0, self.parent.train_capacity * 0.3 - load_per_train)
                total_cost += underutil * self.parent.penalties['underutil']

                frequency = 60 / trains
                wait_time = frequency / 2
                total_cost += wait_time * geo_mean_ons * self.parent.penalties['wait_time']

                if frequency < slot.min_frequency:
                    total_cost += (slot.min_frequency - frequency) * self.parent.penalties['frequency']
                elif frequency > slot.max_frequency:
                    total_cost += (frequency - slot.max_frequency) * self.parent.penalties['frequency']
            else:
                if geo_mean_ons > 0:
                    total_cost += self.parent.penalties['no_trains']

        return total_cost

