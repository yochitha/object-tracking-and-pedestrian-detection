import cv2
import numpy as np
import os

from utils import run_kalman_filter, run_particle_filter

np.random.seed(42) 

# I/O directories
input_dir = "input"
output_dir = "output"

class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([[init_x], [init_y], [0.0], [0.0]])  # state
        self.p_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.002, 0], [0, 0, 0, 0.002]])
        self.f_matrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.h_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.q_matrix = Q
        self.r_matrix = R
        
        return None

    def predict(self):
        self.state = np.dot(self.f_matrix, self.state)
        self.p_matrix = self.f_matrix.dot(self.p_matrix).dot(self.f_matrix.transpose()) + self.q_matrix
        
        return None

    def correct(self, meas_x, meas_y):
        self.z_matrix = np.array([[meas_x], [meas_y]])
        self.y_matrix = np.subtract(self.z_matrix, np.dot(self.h_matrix, self.state))
        self.s_matrix = np.add((self.h_matrix.dot(self.p_matrix).dot(self.h_matrix.transpose())), self.r_matrix)
        self.k_matrix = self.p_matrix.dot(self.h_matrix.transpose()).dot(np.linalg.inv(self.s_matrix))
        self.state = np.add(self.state, np.dot(self.k_matrix, self.y_matrix))
        self.p_matrix = np.subtract(self.p_matrix, (self.p_matrix.dot(self.k_matrix).dot(self.h_matrix)))

        return None

    def process(self, measurement_x, measurement_y):
        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): This should be a N x 2 array where
                                        N = self.num_particles.
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  
        self.sigma_exp = kwargs.get('sigma_exp') 
        self.sigma_dyn = kwargs.get('sigma_dyn')  
        self.template_rect = kwargs.get('template_coords') 

        self.template = template
        self.frame = frame
        self.particles = np.empty((self.num_particles, 2))
        self.particles[:, 0] = np.random.uniform(self.template_rect['x'], self.template_rect['x'] + self.template_rect['w'], size=self.num_particles)
        self.particles[:, 1] = np.random.uniform(self.template_rect['y'], self.template_rect['y'] + self.template_rect['h'], size=self.num_particles)
        # Initialize your particles array. 
        self.weights = np.array([.5] * self.num_particles)  
        # Initialize any other components you may need when designing your filter.
        self.res_particles = []

        return None

    def get_particles(self):
        """Returns the current particles state.
        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.
        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """

        temp_img = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        mse = np.sum((temp_img.astype("float") - frame_cutout.astype("float")) ** 2)
        mse /= float(temp_img.shape[0] * temp_img.shape[1])

        return mse

    def resample_particles(self):
        """Returns a new set of particles

        Returns:
            numpy.array: particles data structure.
        """
        weights_sum = sum(self.weights)

        for i in range(self.num_particles):
            self.weights[i] /= weights_sum

        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)

        self.res_particles = self.particles[indices]

        return self.res_particles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for i in range(self.num_particles):
            self.particles[i][0] += np.random.uniform(-self.sigma_dyn, self.sigma_dyn)
            self.particles[i][1] += np.random.uniform(-self.sigma_dyn, self.sigma_dyn)

            if int(self.particles[i][1]) + self.template.shape[0] // 2 > frame.shape[0] or int(self.particles[i][0]) + self.template.shape[1] // 2 > frame.shape[1] or int(self.particles[i][1]) < self.template.shape[0] // 2 or int(self.particles[i][0]) < self.template.shape[1] // 2:
                self.weights[i] = 0.0
            else:
                frame_cutout = frame[int(self.particles[i][1]) - int(np.ceil(self.template.shape[0] / 2)): int(self.particles[i][1]) + int(np.floor(self.template.shape[0] / 2)),
                                int(self.particles[i][0]) - int(np.ceil(self.template.shape[1] / 2)): int(self.particles[i][0]) + int(np.floor(self.template.shape[1] / 2))]

                error = self.get_error_metric(self.template, frame_cutout)

                self.weights[i] = np.exp(- error / (2 * (self.sigma_exp ** 2)))

        self.particles = self.resample_particles()

        return None

    def render(self, frame_in):
        """Visualizes current particle filter state.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0
        distance = []
        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        for x, y in self.particles.astype(int):
            cv2.circle(frame_in, (x, y), 1, [0, 0, 255], -1)

        cv2.rectangle(frame_in, (int(x_weighted_mean) - self.template.shape[1] // 2, int(y_weighted_mean) - self.template.shape[0] // 2),
                      (int(x_weighted_mean) + self.template.shape[1] // 2, int(y_weighted_mean) + self.template.shape[0] // 2), [255, 0, 0], 2)

        for i in range(self.num_particles):
            distance.append(np.sqrt((self.particles[i, 0] - x_weighted_mean) ** 2 + (self.particles[i, 1] - y_weighted_mean) ** 2))

        w_sum = 0
        for i in range(self.num_particles):
           w_sum += distance[i] * self.weights[i]

        radius = w_sum / sum(self.weights)
        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(radius), [0, 255, 0], 2)

        return None


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        ParticleFilter.process(self, frame)
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        new_frame = frame[int(y_weighted_mean) - int(np.ceil(self.template.shape[0] / 2)): int(y_weighted_mean) + int(np.floor(self.template.shape[0] / 2)),
                                int(x_weighted_mean) - int(np.ceil(self.template.shape[1] / 2)): int(x_weighted_mean) + int(np.floor(self.template.shape[1] / 2))]

        self.template = self.alpha * new_frame + (1 - self.alpha) * self.template
        image = np.zeros(self.template.shape)
        cv2.normalize(self.template, image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        self.template = image
        self.template = np.float32(self.template)

        return None

class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        self.alpha = 0.05

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        ParticleFilter.process(self, frame)
        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

        new_frame = frame[int(y_weighted_mean) - int(np.ceil(self.template.shape[0] / 2)): int(y_weighted_mean) + int(
            np.floor(self.template.shape[0] / 2)),
                    int(x_weighted_mean) - int(np.ceil(self.template.shape[1] / 2)): int(x_weighted_mean) + int(
                        np.floor(self.template.shape[1] / 2))]

        self.template = self.alpha * new_frame + (1 - self.alpha) * self.template

        if self.template.shape[0] > 90 and self.template.shape[1] >= 45:
            self.template = cv2.resize(self.template, (0, 0), fx=0.989, fy=0.99)
        elif self.template.shape[0] == 90 and self.template.shape[1] <= 45:
            self.template = cv2.resize(self.template, (0, 0), fx=0.98, fy=1.0)

        image = np.zeros(self.template.shape)
        cv2.normalize(self.template, image, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        self.template = image
        self.template = np.float32(self.template)

        return None

def part_1b(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_2 = {'x': 7.5, 'y': 7.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_2, "matching",
                            save_frames, template_loc, Q, R)
    return out


def part_1c(obj_class, template_loc, save_frames, input_folder):
    Q = 0.1 * np.eye(4)  # Process noise array
    R = 0.1 * np.eye(2)  # Measurement noise array
    NOISE_1 = {'x': 2.5, 'y': 2.5}
    out = run_kalman_filter(obj_class, input_folder, NOISE_1, "hog",
                            save_frames, template_loc, Q, R)
    return out


def part_2a(obj_class, template_loc, save_frames, input_folder):
    num_particles = 1000  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 10  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_2b(obj_class, template_loc, save_frames, input_folder):
    num_particles = 100  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 15  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        template_loc,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        template_coords=template_loc)  # Add more if you need to
    return out


def part_3(obj_class, template_rect, save_frames, input_folder):
    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 25  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.05  # Set a value for alpha

    out = run_particle_filter(
        obj_class,  # particle filter model class
        input_folder,
        # input video
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out


def part_4(obj_class, template_rect, save_frames, input_folder):
    num_particles = 500  # Define the number of particles
    sigma_md = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 5  # Define the value of sigma for the particles movement (dynamics)

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        template_coords=template_rect)  # Add more if you need to
    return out


def run_particle_filter_part5(filter_class, imgs_dir, template_rect1, template_rect2, save_frames1={},
                        save_frames2={}, **kwargs):

    imgs_list = [f for f in os.listdir(imgs_dir)
                 if f[0] != '.' and f.endswith('.jpg')]
    imgs_list.sort()

    # Initialize objects
    template1 = None
    template2 = None
    pf1 = None
    pf2 = None
    frame_num1 = 0
    frame_num2 = 0
    frames_list = []
    imgs_list1 = imgs_list[27:]
    imgs_list2 = imgs_list[:57]

    # Loop over video (till last frame or Ctrl+C is presssed)
    for img in imgs_list1:

        frame = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template1 is None:
            template1 = frame[int(template_rect1['y']):
                             int(template_rect1['y'] + template_rect1['h']),
                             int(template_rect1['x']):
                             int(template_rect1['x'] + template_rect1['w'])]

            if 'template' in save_frames1:
                cv2.imwrite(save_frames1['template'], template1)

            pf1 = filter_class(frame, template1, **kwargs)
        # Process frame
        pf1.process(frame)

        if True:  # For debugging, it displays every frame
            out_frame = frame.copy()
            pf1.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num1 in save_frames1:
            frame_out = frame.copy()
            pf1.render(frame_out)
            frames_list.append(frame_out)
            cv2.imwrite(save_frames1[frame_num1], frame_out)

        # Update frame number
        frame_num1 += 1
        if frame_num1 % 20 == 0:
            print('Working on frame {}'.format(frame_num1))

    for img in imgs_list2:

        frame1 = cv2.imread(os.path.join(imgs_dir, img))

        # Extract template and initialize (one-time only)
        if template2 is None:
            template2 = frame1[int(template_rect2['y']):
                              int(template_rect2['y'] + template_rect2['h']),
                        int(template_rect2['x']):
                        int(template_rect2['x'] + template_rect2['w'])]

            if 'template' in save_frames2:
                cv2.imwrite(save_frames2['template'], template2)

            kwargs['sigma_dyn'] = 20
            kwargs['template_coords'] = kwargs['template_coords2']
            pf2 = filter_class(frame1, template2, **kwargs)
        # Process frame
        pf2.process(frame1)

        if True:  # For debugging, it displays every frame
            out_frame = frame1.copy()
            pf2.render(out_frame)
            cv2.imshow('Tracking', out_frame)
            cv2.waitKey(1)

        # Render and save output, if indicated
        if frame_num2 in save_frames2:
            frame_out = frame.copy()
            frame_render = frames_list.pop(0)
            pf2.render(frame_render)
            cv2.imwrite(save_frames2[frame_num2], frame_render)

        # Update frame number
        frame_num2 += 1
        if frame_num2 % 20 == 0:
            print('Working on frame {}'.format(frame_num2))
    return 0


def part_5(obj_class, template_loc1, template_loc2, save_frames1, save_frames2, input_folder):
    num_particles = 500  # Define the number of particles
    sigma_mse = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn1 = 15
    sigma_dyn2 = 20
    alpha = 0.03

    out = run_particle_filter_part5(
        obj_class,  # particle filter model class
        input_folder,
        template_loc1,
        template_loc2,
        save_frames1,
        save_frames2,
        num_particles=num_particles,
        sigma_exp=sigma_mse,
        sigma_dyn=sigma_dyn1,
        sigma_dyn1=sigma_dyn2,
        alpha=alpha,
        template_coords=template_loc1,
        template_coords2=template_loc2)  # Add more if you need to
    return out


def part_6(obj_class, template_rect, save_frames, input_folder):
    num_particles = 300  # Define the number of particles
    sigma_md = 10  # Define the value of sigma for the measurement exponential equation
    sigma_dyn = 20  # Define the value of sigma for the particles movement (dynamics)
    alpha = 0.08

    out = run_particle_filter(
        obj_class,
        input_folder,
        template_rect,
        save_frames,
        num_particles=num_particles,
        sigma_exp=sigma_md,
        sigma_dyn=sigma_dyn,
        alpha=alpha,
        template_coords=template_rect)  # Add more if you need to
    return out
