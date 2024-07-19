module RecursiveArrayToolsExt

import RecursiveArrayTools: recursivecopy, recursivecopy!
import MarkovKernels: Normal

recursivecopy(P::Normal) = Normal(recursivecopy(P.μ), recursivecopy(P.Σ))
recursivecopy!(dst::Normal, src::Normal) = begin
    recursivecopy!(dst.μ, src.μ)
    recursivecopy!(dst.Σ, src.Σ)
    return dst
end

end
