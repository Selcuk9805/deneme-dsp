class TimelineService:
    @staticmethod
    def calculate_sync(tempo_a: float, tempo_b: float) -> tuple[float, float, float]:
        """
        Calculates target BPM and speed ratios for A and B.
        Returns: (target_bpm, track_a_speed_ratio, track_b_speed_ratio)
        """
        diff = abs(tempo_a - tempo_b)
        max_tempo = max(tempo_a, tempo_b)
        
        if max_tempo > 0 and diff / max_tempo <= 0.15:
            # Meet in the middle
            target_bpm = (tempo_a + tempo_b) / 2.0
            track_a_ratio = target_bpm / tempo_a
            track_b_ratio = target_bpm / tempo_b
            return target_bpm, track_a_ratio, track_b_ratio
        
        # Don't sync if too far apart or 0
        return tempo_a, 1.0, 1.0

    @staticmethod
    def source_to_execution_time(source_time: float, start_source_time: float, speed_ratio: float) -> float:
        """
        Converts a time in the track's original timeline (source_time) 
        to the SoLoud execution time, assuming playback starts at start_source_time.
        execution_time = (source_time - start_source_time) / speed_ratio
        """
        if speed_ratio <= 0.0:
            speed_ratio = 1.0
            
        elapsed_source = source_time - start_source_time
        return elapsed_source / speed_ratio
