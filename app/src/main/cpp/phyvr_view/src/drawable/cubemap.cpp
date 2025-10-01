//
// Created by samuel on 19/03/2023.
//

#include <phyvr_view/cubemap.h>

#include <phyvr_utils/cache.h>
#include <phyvr_utils/singleton.h>

#include <phyvr_view/errors.h>
#include <phyvr_view/program.h>

CubeMap::CubeMap(const std::shared_ptr<AbstractFileReader> &file_reader,
                 const std::string &pngs_root_path) {

  std::vector<float> vertices{
      -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f, -1.f,
      1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,  -1.f,

      -1.f, -1.f, 1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  -1.f,
      -1.f, 1.f,  -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f, 1.f,

      1.f,  -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f,  1.f,
      1.f,  1.f,  1.f,  1.f,  1.f,  -1.f, 1.f,  -1.f, -1.f,

      -1.f, -1.f, 1.f,  -1.f, 1.f,  1.f,  1.f,  1.f,  1.f,
      1.f,  1.f,  1.f,  1.f,  -1.f, 1.f,  -1.f, -1.f, 1.f,

      -1.f, 1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  1.f,  1.f,
      1.f,  1.f,  1.f,  -1.f, 1.f,  1.f,  -1.f, 1.f,  -1.f,

      -1.f, -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, -1.f,
      1.f,  -1.f, -1.f, -1.f, -1.f, 1.f,  1.f,  -1.f, 1.f};

  nb_vertices = int(vertices.size() / 3);

  auto cache = Singleton<Cache<std::shared_ptr<Program>>>::get_singleton();
  if (cache->exists(pngs_root_path)) {
    program = cache->get(pngs_root_path);
    return;
  }

  program = Program::Builder(file_reader, "shaders/cube_vs.glsl",
                             "shaders/cube_fs.glsl")
                .add_cube_texture("u_cube_map", pngs_root_path)
                .add_uniform("u_mvp_matrix")
                .add_attribute("a_vp")
                .add_buffer("cube", vertices)
                .build();

  cache->add(pngs_root_path, program);
}

void CubeMap::draw(glm::mat4 mvp_matrix, glm::mat4 mv_matrix,
                   glm::vec3 light_pos_from_camera, glm::vec3 camera_pos) {
  program->use();

  program->uniform_mat4("u_mvp_matrix", mvp_matrix);
  program->cube_texture("u_cube_map");
  program->attrib("a_vp", "cube", POSITION_SIZE, STRIDE, 0);

  Program::draw_arrays(GL_TRIANGLES, 0, nb_vertices);

  program->disable_attrib_array();
  Program::disable_cube_texture();
}

CubeMap::~CubeMap() { program.reset(); }
