#pragma once

#ifdef WITH_YAML_CPP

struct YamlOutputMaker {
    std::string block;
    YamlOutputMaker(const std::string &_block) : block(_block) {}
    std::map<const std::string, double> kv;
    void add(const std::string &key, double value) { kv[key] = value; }
    void add(int key, double value) { add(std::to_string(key), value); }
    void make_output(YAML::Emitter &yaml_out) const {
        yaml_out << YAML::Key << block << YAML::Value;
        yaml_out << YAML::Flow << YAML::BeginMap;
        for (auto &item : kv) {
            yaml_out << YAML::Key << YAML::Flow << item.first 
                     << YAML::Value << item.second;
        }
        yaml_out << YAML::Flow << YAML::EndMap;
    }
};

static void WriteOutYaml(YAML::Emitter &yaml_out, const std::string &bname,
                        const std::vector<YamlOutputMaker> &makers) {
    yaml_out << YAML::Key << YAML::Flow << bname << YAML::Value;
    yaml_out << YAML::Flow << YAML::BeginMap;
    for (auto &m : makers) {
        m.make_output(yaml_out);
    }
    yaml_out << YAML::Flow << YAML::EndMap;
}

#endif
