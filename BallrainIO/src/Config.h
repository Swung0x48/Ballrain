#pragma once

#include <functional>
#include <unordered_map>
#include <BML/IMod.h>

class BallrainConfig {
public:
    BallrainConfig(std::function<IConfig*()> getter): m_getter(getter) {}

    template <typename T>
    IProperty* CreateProperty(const char* category, const char* key, const T& default_v, const char* comment = "") {
        auto prop = m_getter()->GetProperty(category, key);
        prop->SetComment(comment);

        if constexpr (std::is_same_v<T, bool>)
            prop->SetDefaultBoolean(default_v);
        else if constexpr (std::is_same_v<T, int>)
            prop->SetDefaultInteger(default_v);
        else if constexpr (std::is_same_v<T, float>)
            prop->SetDefaultFloat(default_v);
        else if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>)
            prop->SetDefaultString(default_v);

        m_props[key] = prop;
        return prop;
    }

    // used for more complex access instead of GetValue/SetValue
    IProperty* GetProperty(const char* key) {
        if (auto it = m_props.find(key); it != m_props.end())
            return it->second;
        return nullptr;
    }

    template <typename T>
    T Get(const char* key) {
        auto prop = GetProperty(key);
        if constexpr (std::is_same_v<T, bool>)
            return prop->GetBoolean();
        else if constexpr (std::is_same_v<T, int>)
            return prop->GetInteger();
        else if constexpr (std::is_same_v<T, float>)
            return prop->GetFloat();
        else if constexpr (std::is_same_v<T, std::string>)
            return std::string(prop->GetString());
        else if constexpr (std::is_same_v<T, const char*>)
            return prop->GetString();
        else
            static_assert("Unsupported type used for BallrainConfig::Get");
    }

    template <typename T>
    void Set(const char* key, const T& value) {
        auto prop = GetProperty(key);
        if constexpr (std::is_same_v<T, bool>)
            prop->SetBoolean(value);
        else if constexpr (std::is_same_v<T, int>)
            prop->SetInteger(value);
        else if constexpr (std::is_same_v<T, float>)
            prop->SetFloat(value);
        else if constexpr (std::is_same_v<T, const char*> || std::is_same_v<T, char*>)
            prop->SetString(value);
        else if constexpr (std::is_same_v<T, std::string>)
            prop->SetString(value.c_str());
        else
            static_assert("Unsupported type used for BallrainConfig::Set");
    }

    ~BallrainConfig() {
        m_props.clear();
    }

private:
    std::function<IConfig*()> m_getter;
    std::unordered_map<std::string, IProperty*> m_props;
};
