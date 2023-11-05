--- === com.pravdomil.brain2film ===

local fcp = require("cp.apple.finalcutpro")

local plugin = {
    id = "com.pravdomil.brain2film",
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
        :add("Brain2Film")
        :whenActivated(run)
end

function run()
    fcp:launch()

    local info = fcp.inspector.info
    info:show()

    local filename = emptyToNil(info.fileName():value()) or emptyToNil(info.displayName():value()) or ""

    local data = {
        "01HEGNTBG1SFDYQN1WK843952H",
        fcp:activeLibraryPaths(),
        filename,
        info.name():value(),
        info.clipStart():value(),
        info.clipDuration():value(),
        info.notes():value()
    }

    local file = io.popen("/usr/local/bin/python3 ../Plugins/Brain2Film/src/commandpost.py \"$(cat -)\" &", "w")
    file:write(hs.json.encode(data))
    file:close()
end

function emptyToNil(a)
    if a == "" then
        return nil
    else
        return a
    end
end

return plugin
