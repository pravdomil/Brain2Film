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
    hs.alert.show("AI Cut Ultra here!")
end

return plugin
