# define normal parametrisations
abstract type AbstractNormalParametrisation end
struct Usual <: AbstractNormalParametrisation  end
struct Information <: AbstractNormalParametrisation end
struct Precision <: AbstractNormalParametrisation end