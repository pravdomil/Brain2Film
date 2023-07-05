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

    local filename = emptyToNil(info.fileName():value()) or emptyToNil(info.displayName():value()) or ""

    local data = {
        "_dx2rgq3ln9kfsl_wdv9vzlng",
        fcp:activeLibraryPaths(),
        filename,
        info.name():value(),
        info.clipStart():value(),
        info.clipDuration():value(),
        info.notes():value()
    }

    local file = io.popen("python3 ../Plugins/AI\\ Cut\\ Ultra/CommandPost/script.py \"$(cat -)\" &", "w")
    file:write(hs.json.encode(data))
    file:close()

    hs.alert.show("Processing " .. filename)
end

function emptyToNil(a)
    if a == "" then
        return nil
    else
        return a
    end
end

return plugin
