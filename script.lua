--- === com.pravdomil.ai-cut-ultra ===

local fcp = require("cp.apple.finalcutpro")

local plugin = {
    id = "com.pravdomil.ai-cut-ultra",
    group = "finalcutpro",
    dependencies = {
        ["finalcutpro.commands"] = "fcpxCmds",
    }
}

function plugin.init(deps)
    if not fcp:isSupported() then
        return
    end

    deps.fcpxCmds
        :add("AI Cut Ultra")
        :whenActivated(run)
end

function run()
    fcp:launch()

    local info = fcp.inspector.info
    info:show()

    local paths = fcp:activeLibraryPaths()
    local filename = emptyToNil(info.filename():value()) or emptyToNil(info.displayName():value()) or ""
    local notes = info.notes():value()

    local data = { "_dx2rgq3ln9kfsl_wdv9vzlng", paths, filename, notes }

    local file = io.popen("../Plugins/AI\\ Cut\\ Ultra/script.sh", "w")
    file:write(hs.json.encode(data))
    file:close()

    hs.alert.show("AI Cut Ultra!")
end

function emptyToNil(a)
    if a == "" then
        return nil
    else
        return a
    end
end

return plugin