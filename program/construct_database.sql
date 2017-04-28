-- -- build the full history table
-- CREATE TABLE full_history (
-- 	id 					SERIAL 	  		NOT NULL,
--   	timestamp 			VARCHAR(100) 	NOT NULL,
--   	vehicle_id 			INT 			NOT NULL,
--   	latitude 			NUMERIC 		NOT NULL,
-- 	longitude 			NUMERIC 		NOT NULL, 
-- 	bearing 			NUMERIC 		NOT NULL,
-- 	progress 			INT 			NOT NULL,
-- 	service_date 		INT 			NOT NULL,
-- 	trip_id 			VARCHAR(100) 	NOT NULL,
-- 	block_assgined 		INT 			NOT NULL,
-- 	next_stop_id 		INT 			NOT NULL,
-- 	dist_along_route 	NUMERIC 		NOT NULL,
-- 	dist_from_stop 		NUMERIC			NOT NULL,
-- 	total_distance 		NUMERIC 		NOT NULL,
-- 	route_id 			VARCHAR(100) 	NOT NULL,
-- 	shape_id 			VARCHAR(100) 	NOT NULL,
-- 	PRIMARY KEY (id)
-- );

-- COPY full_history
-- FROM '/Users/junwang/PycharmProjects/bus_arrival_prediction/program/preprocessed_full_history.csv' DELIMITER ',' CSV HEADER;

-- -- build the trips table

-- CREATE TABLE trips
-- (
-- 	id				SERIAL			NOT NULL,
-- 	route_id		VARCHAR(100)	NOT NULL,
-- 	service_id		VARCHAR(100)	NOT NULL,
--  	trip_id			VARCHAR(100) 	NOT NULL,
--  	trip_headsign	VARCHAR(100) 	NOT NULL,
--  	direction_id	INT				NOT NULL,
--  	shape_id		VARCHAR(100) 	NOT NULL,
-- 	PRIMARY KEY (id),
-- 	CONSTRAINT trips_unique UNIQUE (trip_id)
-- );

-- COPY trips
-- FROM '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/GTFS/gtfs/trips.txt' DELIMITER ',';

-- CREATE TABLE stop_times
-- (
-- 	id				SERIAL			NOT NULL,
-- 	trip_id			VARCHAR(100) 	NOT NULL,
--  	arrival_time	VARCHAR(100) 	NOT NULL,
--  	departure_time	VARCHAR(100) 	NOT NULL,
--  	stop_id			INT 			NOT NULL,
--  	stop_sequence	INT 			NOT NULL,
--  	pickup_type		INT 			NOT NULL,
-- 	drop_off_type	INT 			NOT NULL,
-- 	PRIMARY KEY(id)
-- );

-- COPY stop_times
-- FROM '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/GTFS/gtfs/stop_times.txt' DELIMITER ',';

-- CREATE TABLE stops
-- (
-- 	id				SERIAL			NOT NULL,
-- 	stop_id  		INT 			NOT NULL,
-- 	stop_name		VARCHAR(100) 	NOT NULL,
-- 	stop_lat  		NUMERIC 		NOT NULL,
-- 	stop_lon      	NUMERIC 		NOT NULL,
-- 	location_type	INT 			NOT NULL,
-- 	PRIMARY KEY(id),
-- 	CONSTRAINT stops_unique UNIQUE (stop_id)
-- );

-- COPY stops(stop_id, stop_name, stop_lat, stop_lon, location_type)
-- FROM '/Users/junwang/PycharmProjects/bus_arrival_prediction/data/GTFS/gtfs/stops.txt' DELIMITER ',';


-- CREATE TABLE route_stop_dist
-- (
-- 	id					SERIAL			NOT NULL,
-- 	stop_id         	INT 			NOT NULL,
-- 	route_id        	VARCHAR(100)	NOT NULL,
-- 	shape_id        	VARCHAR(100)	NOT NULL,
-- 	direction_id    	INT				NOT NULL,
-- 	dist_along_route 	NUMERIC 		NOT NULL,
-- 	PRIMARY KEY (id)
-- );

-- COPY route_stop_dist(stop_id, route_id, shape_id, direction_id, dist_along_route)
-- FROM '/Users/junwang/PycharmProjects/bus_arrival_prediction/program/route_stop_dist.csv' DELIMITER ',' CSV HEADER;

-- CREATE TABLE weather
-- (
-- 	id					SERIAL			NOT NULL,
-- 	date          		INT 			NOT NULL,
-- 	rain  				INT 			NOT NULL,
-- 	snow   				INT 			NOT NULL,
-- 	weather     		INT 			NOT NULL,
-- 	PRIMARY KEY (id)
-- );

-- COPY route_stop_dist(date, rain, snow, weather)
-- FROM '/Users/junwang/PycharmProjects/bus_arrival_prediction/program/weather.csv' DELIMITER ',' CSV HEADER;


ALTER TABLE trips ADD COLUMN id SERIAL PRIMARY KEY NOT NULL;
ALTER TABLE stop_times ADD COLUMN id SERIAL PRIMARY KEY NOT NULL;
ALTER TABLE stops ADD COLUMN id SERIAL PRIMARY KEY NOT NULL;