#include <vector>
#include <math.h>
#include <time.h>
#include <random>
#include <ANN/ANN.h>
#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include <QtGui/QLabel>
#include <QtGui/QImage>

using namespace std;

//----------------------------------
//--------------Macros--------------
//----------------------------------
template<typename T>
T mix(const T &a, const T &b, const T &mix)
{
	return b * mix + a * (T(1) - mix);
}

//-------------------------------------
//--------------Constants--------------
//-------------------------------------
#define PI 3.141592653589793
#define INFINITY 1e8

//--------------------------------------
//--------------Parameters--------------
//--------------------------------------
#define PHOTON_MAPPING			// define for Photon Mapping, undefine for Ray Tracing
#define WINDOW_WIDTH 512		// Window width
#define WINDOW_HEIGHT 512		// Window height
#define MAX_RAY_DEPTH 5			// Max recursive depth
#define PHOTONS 500				// Number of photons
#define CAUSTICS_PHOTONS 20000	// Number of photons for caustic objects
#define LIGHT_POWER 500			// Lights power
#define ESTIMATE 100					// Number of nearest neighbors
#define CAUSTICS_ESTIMATE 500
#define EXPOSURE 11

//-----------------------------------
//--------------Vector3--------------
//-----------------------------------
template<typename T>
class Vec3
{
public:
	T x, y, z;
	Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
	Vec3(T xx) : x(xx), y(xx), z(xx) {}
	Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	Vec3& normalize()
	{
		T nor2 = length2();
		if (nor2 > 0) {
			T invNor = 1 / sqrt(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}

	// clam values under/above min/max
	Vec3& gate(T min, T max)
	{
		if (x < min)
			x = min; 
		if (y < min)
			y = min;
		if (z < min)
			z = min;

		if (x > max)
			x = max; 
		if (y > max)
			y = max;
		if (z > max)
			z = max;

		return *this;
	}
	Vec3<T> operator * (const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
	Vec3<T> operator * (const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
	Vec3<T> operator / (const T &f) const { return Vec3<T>(x / f, y / f, z / f); }
	Vec3<T> operator / (const Vec3<T> &v) const { return Vec3<T>(x / v.x, y / v.y, z / v.z); }
	T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
	Vec3<T> cross(const Vec3<T> &v) const
	{
		Vec3<T> res;
		res.x = this->y*v.z - this->z*v.y;
		res.y = this->z*v.x - this->x*v.z;
		res.z = this->x*v.y - this->y*v.x;
		return res;
	}
	Vec3<T> operator - (const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
	Vec3<T> operator + (const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
	Vec3<T>& operator += (const Vec3<T> &v) { x += v.x, y += v.y, z += v.z; return *this; }
	Vec3<T>& operator *= (const Vec3<T> &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }
	Vec3<T>& operator /= (const Vec3<T> &v) { x /= v.x, y /= v.y, z /= v.z; return *this; }
	Vec3<T> operator - () const { return Vec3<T>(-x, -y, -z); }
	T length2() const { return x * x + y * y + z * z; }
	T length() const { return sqrt(length2()); }

	friend std::ostream & operator << (std::ostream &os, const Vec3<T> &v)
	{
		os << "[" << v.x << " " << v.y << " " << v.z << "]";
		return os;
	}
};

//---------------------------------------
//--------------Geom Object--------------
//---------------------------------------
template<typename T>
class GeomObject {

public:
	Vec3<T> surfaceColor, emissionColor;    /// surface color and emission (light)
	T transparency, reflection;             /// surface transparency and reflectivity

	GeomObject(const Vec3<T> &sc, const T &refl = 0, const T &transp = 0, const Vec3<T> &ec = 0)
		: surfaceColor(sc), reflection(refl), transparency(transp), emissionColor(ec)
	{}

	// get object position
	virtual Vec3<T> getPosition() = 0;
	virtual bool intersect(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, Vec3<T>* pHit = NULL, Vec3<T>* nHit = NULL) const = 0;
	virtual Vec3<T> computeBRDF() const = 0;
	virtual Vec3<T> randomPoint() const = 0;
};

//---------------------------------
//--------------Plane--------------
//---------------------------------
template<typename T>
class Plane : public GeomObject<T> {

public:
	Vec3<T> position;		// plane position
	Vec3<T> normal;		// vector normal to the plane

	Plane(const Vec3<T> &p, const Vec3<T> &n, const Vec3<T> &sc,
		const T &refl = 0, const T &transp = 0, const Vec3<T> &ec = 0) :
		position(p), normal(n), GeomObject(sc, refl, transp, ec)
	{}

	// get plane position
	Vec3<T> getPosition() { return position; }

	// ray-plane intersection
	bool intersect(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, Vec3<T>* pHit = NULL, Vec3<T>* nHit = NULL) const
	{
		Vec3<T> negNorm = -normal;
		T denom = negNorm.dot(rayDir);
		if (denom > 1e-6)
		{
			Vec3<T> aux = position - rayOrig;
			T d = aux.dot(negNorm) / denom;
			
			if (d >= 0)
			{
				if (pHit != NULL && nHit != NULL)
				{
					*pHit = rayOrig + rayDir*d;
					*nHit = normal;
				}

				return true;
			}
		}

		return false;
	}

	Vec3<T> computeBRDF() const
	{
		// TODO: BRDF for glossy objects
		return surfaceColor; // since it is a diffuse object, simply returns surface color
	}

	Vec3<T> randomPoint() const
	{
		return Vec3<T>(0);
	}
};
//----------------------------------
//--------------Sphere--------------
//----------------------------------
template<typename T>
class Sphere : public GeomObject<T> {

public:
	Vec3<T> center;                         /// position of the sphere
	T radius, radius2;                      /// sphere radius and radius^2
	
	Sphere(const Vec3<T> &c, const T &r, const Vec3<T> &sc, 
		const T &refl = 0, const T &transp = 0, const Vec3<T> &ec = 0) : 
		center(c), radius(r), radius2(r * r), GeomObject(sc, refl, transp, ec)
	{}

	Vec3<T> getPosition() { return center; }

	// compute a ray-sphere intersection using the geometric solution
	// http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-7-intersecting-simple-shapes/ray-sphere-intersection/
	bool intersect(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, Vec3<T>* pHit = NULL, Vec3<T>* nHit = NULL) const
	{
		T t0;
		T t1;
		Vec3<T> l = center - rayOrig;
		T tca = l.dot(rayDir);
		if (tca < 0) return false;
		T d2 = l.dot(l) - tca * tca;
		if (d2 > radius2) return false;
		T thc = sqrt(radius2 - d2);
		
		t0 = tca - thc;
		t1 = tca + thc;

		if (t0 < 0) t0 = t1;

		if (pHit != NULL && nHit != NULL)
		{
			*pHit = rayOrig + rayDir * t0; // point of intersection
			*nHit = *pHit - center; // normal at the intersection point
			(*nHit).normalize(); // normalize normal direction
		}

		return true;
	}

	Vec3<T> computeBRDF() const
	{
		// TODO: BRDF for glossy objects
		return surfaceColor; // since it is a diffuse object, simply returns surface color
	}

	Vec3<T> randomPoint() const
	{
		T x;
		T y;
		T z;
		
		// random vector in unit sphere
		do
		{
			x = ((T) rand() / (RAND_MAX))*2 - 1;			// random coordinates between -1 and 1
			y = ((T) rand() / (RAND_MAX))*2 - 1;
			z = ((T) rand() / (RAND_MAX))*2 - 1;
		} while(pow(x,2) + pow(y,2) + pow(z,2) > 1);		// simple rejection sampling

		return center + Vec3<T>(x,y,z) * radius;
	}
};

//--------------------------------------------
//--------------Global Variables--------------
//--------------------------------------------
Vec3<float> g_origin = Vec3<float>(0,0,0);
vector<GeomObject<float>*> g_objects;		// objects in the scene
vector<GeomObject<float>*> g_lights;		// lights
vector<GeomObject<float>*> g_caustics;		// objects which can produce caustics
QWidget* g_window;							// rendering with QT
QImage* g_image;
QLabel* g_label; 

//----------------------------------
//--------------Camera--------------
//----------------------------------
template<typename T>
class Camera {

public:
	Vec3<T> position;
	Vec3<T> lookAt;
	Vec3<T> up;
	Vec3<T> right;

	Camera(Vec3<T> &p, Vec3<T> &l, Vec3<T> &u, Vec3<T> &r)
		: position(p), lookAt(l), up(u), right(r)
	{}

	// transforms a pixel to world coordinates
	// params: pixel x, pixel y, offset x, offset y
	Vec3<T> pixelToWorld(int i, int j, T offset_x, T offset_y)
	{
		T remapped_x = (2*((i + offset_x)/WINDOW_WIDTH) - 1)*(WINDOW_WIDTH/WINDOW_HEIGHT);
		T remapped_y = 1 - 2*((j + offset_y)/WINDOW_HEIGHT);

		Vec3<T> w = position - lookAt;
		w.normalize();
		Vec3<T> u = up.cross(w);
		u.normalize();
		Vec3<T> v = w.cross(u);
		v.normalize();

		Vec3<T> transf = -u*remapped_x + v*remapped_y - w;
		transf.normalize();

		return position + transf;
	}

	// Given a point in 3D space returns the ortographic projection
	// of the point to the plane x / y
	Vec3<T> worldToPixel(Vec3<T> p)
	{
		/*// create orthonormal base for the camera
		Vec3<T> w = position - lookAt;
		w.normalize();
		Vec3<T> u = up.cross(w);
		u.normalize();
		Vec3<T> v = w.cross(u);
		v.normalize();

		Vec3<T> z = w * ((p-position).dot(w));
		Vec3<T> y = (p-position) - z;

		Vec3<T> ys = (y/z)*(-1);		//projection onto near plane

		// remap back to pixel coordinates*/

		// drop z coordinate and remap from [-20, 20] to [0, window_dim]
		return Vec3<T>(((WINDOW_WIDTH-1)*(p.x+20))/40, ((WINDOW_HEIGHT-1)*(p.y+20))/40, 0);
		
	}
};

//-------------------------------------
//--------------Raytracer--------------
//-------------------------------------
template<typename T>
class RayTracer {

public:

	RayTracer() {}

	Vec3<T> trace(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, const int &depth)
	{
		T minDist = INFINITY;				// minimum distance found
		Vec3<T> pHit;						// point of intersection
		Vec3<T> nHit;						// normal at the intersection point
		const GeomObject<T> *obj = NULL;	// closest object

		// find intersection of this ray with the objects in the scene
		for (unsigned i = 0; i < g_objects.size(); ++i)
		{
			Vec3<T> p;
			Vec3<T> n;
			if (g_objects[i]->intersect(rayOrig, rayDir, &p, &n))
			{
				T dist = (p - rayOrig).length();
				if (dist < minDist)
				{
					minDist = dist;
					obj = g_objects[i];
					pHit = p;
					nHit = n;
				}
			}
		}

		// if there's no intersection return black or background color
		if (!obj) return Vec3<T>(0);
		Vec3<T> surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray

		nHit.normalize(); // normalize normal direction

		// If the normal and the view direction are not opposite to each other 
		// reverse the normal direction. That also means we are inside the sphere so set
		// the inside bool to true. Finally reverse the sign of IdotN which we want
		// positive.
		T bias = 1e-4; // add some bias to the point from which we will be tracing
		bool inside = false;
		if (rayDir.dot(nHit) > 0) nHit = -nHit, inside = true;
		if ((obj->transparency > 0 || obj->reflection > 0) && depth < MAX_RAY_DEPTH)
		{
			T facingratio = -rayDir.dot(nHit);
			// change the mix value to tweak the effect
			T fresneleffect = mix<T>(pow(1 - facingratio, 3), 1, 0.1); 
			// compute reflection direction (not need to normalize because all vectors
			// are already normalized)
			Vec3<T> refldir = rayDir - nHit * 2 * rayDir.dot(nHit);
			refldir.normalize();
			Vec3<T> reflection = trace(pHit + nHit * bias, refldir, depth + 1);

			Vec3<T> refraction = 0;
			// if the sphere is also transparent compute refraction ray (transmission)
			if (obj->transparency) {
				T ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
				T cosi = -nHit.dot(rayDir);
				T k = 1 - eta * eta * (1 - cosi * cosi);
				Vec3<T> refrdir = rayDir * eta + nHit * (eta *  cosi - sqrt(k));
				refrdir.normalize();
				refraction = trace(pHit - nHit * bias, refrdir, depth + 1);
			}
			// the result is a mix of reflection and refraction (if the sphere is transparent)
			surfaceColor = (reflection * fresneleffect * obj->reflection + 
				refraction * (1 - fresneleffect) * obj->transparency) * obj->surfaceColor;
		}
		else 
		{
			// it's a diffuse object, no need to raytrace any further
			for (unsigned i = 0; i < g_lights.size(); ++i)
			{
				// this is a light
				Vec3<T> transmission = 1;
				Vec3<T> lightDirection = g_lights[i]->getPosition() - pHit;
				lightDirection.normalize();
				for (unsigned j = 0; j < g_objects.size(); ++j)
				{
					if (g_objects[j] == g_lights[i])
						continue;

					Vec3<T> p;
					Vec3<T> n;
					if (g_objects[j]->intersect(pHit + nHit * bias, lightDirection, &p, &n))
					{
						// check if the object is closer than the light
						if ((p-pHit).length() < (g_lights[i]->getPosition()-pHit).length())
						{
							transmission = 0;
							break;
						}
					}
				}
				
				surfaceColor += obj->surfaceColor * transmission * 
					std::max(T(0), nHit.dot(lightDirection)) * g_lights[i]->emissionColor;
			}
		}

		return surfaceColor + obj->emissionColor;
	}
};

//-----------------------------------
//--------------Structs--------------
//-----------------------------------
struct Photon {
    float x, y, z;				// position ( 3 x 32 bit floats )
    Vec3<float> power;			// power(rgb) packed as 3 float
    float phi, theta;			// compressed incident direction
    short flag;					// flag used for kd-tree
};

//-----------------------------------------
//--------------Photon Mapper--------------
//-----------------------------------------
template<typename T>
class PhotonMapper {

private:

	Vec3<T> randomPointInUnitSphere()
	{
		T x;
		T y;
		T z;
		
		// random vector in unit sphere
		do
		{
			x = ((T) rand() / (RAND_MAX))*2 - 1;			// random coordinates between -1 and 1
			y = ((T) rand() / (RAND_MAX))*2 - 1;
			z = ((T) rand() / (RAND_MAX))*2 - 1;
		} while(pow(x,2) + pow(y,2) + pow(z,2) > 1);		// simple rejection sampling

		return Vec3<T>(x,y,z);
	}

	// Reflection (Snell law)
	Vec3<T> specularDir(const Vec3<T> &v, const Vec3<T> &n) { return v - n*2*v.dot(n); }

	// Random diffuse direction (random direction in unit hemisphere)
	Vec3<T> diffuseDir(const Vec3<T> &v, const Vec3<T> &n)
	{
		Vec3<T> rand = randomPointInUnitSphere();
		rand.normalize();
		T sign = rand.dot(n) / abs(rand.dot(n));
		return rand * (rand.dot(n) / abs(rand.dot(n)));
	}

	Vec3<T> refractionDir(const Vec3<T> &rayDir, Vec3<T> &nHit)
	{
		Vec3<T> refrdir;
		//Vec3<T> nHit = n;
		bool inside = false;
		if (rayDir.dot(nHit) > 0) nHit = -nHit, inside = true;

		T ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
		T cosi = -nHit.dot(rayDir);
		T k = 1 - eta * eta * (1 - cosi * cosi);
		refrdir = rayDir * eta + nHit * (eta *  cosi - sqrt(k));
		refrdir.normalize();
		
		return refrdir;
	}

	// trace a photon through the scene
	void tracePhoton(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, Photon &p, const int &depth, bool caustic)
	{
		if (depth > MAX_RAY_DEPTH)
			return;

		T minDist = INFINITY;				// minimum distance found
		Vec3<T> pHit;						// point of intersection
		Vec3<T> nHit;						// normal at the intersection point
		GeomObject<T> *obj = NULL;	// closest object

		if (!checkIntersection(rayOrig, rayDir, obj, pHit, nHit))	// closest object
			return;

		// set photon position
		p.x = pHit.x;
		p.y = pHit.y;
		p.z = pHit.z;
		
		// normalize
		nHit.normalize();

		// dx, dy TODO: VERIFY!
		// http://en.wikipedia.org/wiki/Vector_projection
		T dx = rayDir.dot(nHit);								// projection x of rayDir onto nHit
		T dy = (rayDir - nHit*dx).length();						// projection y of rayDir onto nHit

		float acs = acos(dx);
		p.phi = 255 * (atan2(dy,dx) + PI) / (T)(2*PI);			// compute phi angle
		p.theta = 255 * acos(dx) / (T)PI;						// compute theta angle

		// russian roulette: determine if the current photon
		// is Reflected, Refracted or Absorbed
		T pRefl = max(max(obj->surfaceColor.x + obj->reflection, obj->surfaceColor.y + obj->reflection),
			obj->surfaceColor.z + obj->reflection)/2;
		T diffSum = obj->surfaceColor.x + obj->surfaceColor.y + obj->surfaceColor.z;
		T pDiff = (diffSum/(diffSum + 3*obj->reflection)) * pRefl;								// Diffuse Probability
		T pSpec = pRefl - pDiff;														// Specular Probability

		//Reflected, Refracted or Absorbed?
		T dice = ((T) rand() / (RAND_MAX));										//Random between 0 and 1
		T bias = 1e-4;							// add some bias to the point from which we will be tracing

		if (caustic && obj->transparency > 0)
		{
			Vec3<T> refrdir = refractionDir(rayDir,nHit);
			tracePhoton(pHit - nHit*bias, refrdir, p, depth+1, caustic);
		}

		if (dice >= 0 && dice < pDiff)													// diffuse reflection
		{
			// modify photon power accordingly
			//p.power *= obj->surfaceColor * pDiff;

			Photon cpy = p;
			cpy.power *= obj->surfaceColor * pDiff;

			if (obj->transparency == 0)
			{
				if (caustic)
					storeCausticPhoton(p);
				else
					storePhoton(cpy);
			}
			
			tracePhoton(pHit + nHit*bias, specularDir(pHit,nHit), p, depth+1, caustic);
		}
		else if (dice >= pDiff && dice < pDiff + pSpec)					// specular reflection
		{
			// modify photon power accordingly
			//p.power *= obj->reflection * pSpec;

			tracePhoton(pHit + nHit*bias, specularDir(pHit,nHit), p, depth+1, caustic);
		}
		else if (dice >= pDiff + pSpec && dice < 1)			// absorption
		{
			Photon cpy = p;
			cpy.power *= obj->surfaceColor * (1 - pRefl);
			
			if (obj->transparency == 0)
			{
				if (caustic)
					storeCausticPhoton(p);
				else
					storePhoton(cpy);
			}
			
			return; // photon is absorbed
		}

		// check for shadow photons
		if (depth == 0 && !caustic)
		{
			Vec3<T> new_rayOrig = pHit;
			Vec3<T> temp_p;
			Vec3<T> temp_n;
			GeomObject<T>* o = NULL;
			
			while(checkIntersection(new_rayOrig + rayDir*bias, rayDir, o, temp_p, temp_n) && 
				  o == obj)	// closest object
			{
				o = NULL;
				new_rayOrig = temp_p;
			}

			if (temp_p.x >= -20 && temp_p.x <= 20 &&
				temp_p.y >= -20 && temp_p.y <= 20 &&
				temp_p.z >= -20 && temp_p.z <= 20)
			{
				Photon shadowPhoton;
				shadowPhoton.power = Vec3<T>((-0.03) * (1-obj->transparency));
				shadowPhoton.x = temp_p.x;
				shadowPhoton.y = temp_p.y;
				shadowPhoton.z = temp_p.z;

				storePhoton(shadowPhoton);
			}
		}
	}

	// store photon in the global photon map
	void storePhoton(Photon &p)
	{
		photonMap.push_back(p);		// store in photon list
	}

	// store photon in caustic photon map
	void storeCausticPhoton(Photon &p)
	{
		causticsMap.push_back(p);		// store in photon list
	}

	bool checkIntersection(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, GeomObject<T>* &obj, Vec3<T> &pHit, Vec3<T> &nHit)
	{
		T minDist = INFINITY;				// minimum distance found

		// find intersection of this ray with the objects in the scene
		for (unsigned i = 0; i < g_objects.size(); ++i)
		{
			Vec3<T> p;
			Vec3<T> n;
			if (g_objects[i]->intersect(rayOrig, rayDir, &p, &n))
			{
				T dist = (p - rayOrig).length();
				if (dist < minDist && dist > 0)
				{
					minDist = dist;
					obj = g_objects[i];
					pHit = p;
					nHit = n;
				}
			}
		}

		if (obj != NULL)
			return true;

		return false;
	}

	// compute radiance (using NN algorithm)
	Vec3<T> computeRadiance(const Vec3<T> p, GeomObject<T>* &obj)
	{
		ANNpoint queryPt;						// query point
		ANNpoint caustics_queryPt;				// caustics query point

		// gather photons around pHit
		queryPt = annAllocPt(3);				// allocate query point
		queryPt[0] = p.x;
		queryPt[1] = p.y;
		queryPt[2] = p.z;
		kdTree->annkSearch(						// search
				queryPt,						// query point
				ESTIMATE,						// number of near neighbors
				nnIdx,							// nearest neighbors (returned)
				dists,							// distance (returned)
				0);								// error bound

		// compute SUM(BRDF * PhotonPowers) around pHit
		T maxDist = 0;
		Vec3<T> sum;

		// find max distance
		for (unsigned i = 0; i < ESTIMATE; i++)
		{
			if (maxDist < dists[i])
				maxDist = dists[i];
		}

		T k = 1;
		for (unsigned i = 0; i < ESTIMATE; i++)
		{
			// Gaussian filter
			//T w = (0.918)*(1 - ((1 - exp(-(1.953)*(pow(dists[i],2)/(2*maxDist))))/(1 - exp(-1.953))));
			
			T w = 1 - dists[i]/(k*maxDist);
			//T w = 1;
			sum += photonMap[nnIdx[i]].power * w * obj->computeBRDF();
		}

		//-------------------------------------------------
		//-------------------CAUSTIC-----------------------
		//-------------------------------------------------
		if (g_caustics.size() == 0)
		{
			sum /= (T)PI*maxDist*(1-2/(k*3));
			return sum;
		}

		// gather photons around pHit
		caustics_queryPt = annAllocPt(3);				// allocate query point
		caustics_queryPt[0] = p.x;
		caustics_queryPt[1] = p.y;
		caustics_queryPt[2] = p.z;
		caustics_kdTree->annkSearch(						// search
				caustics_queryPt,						// query point
				CAUSTICS_ESTIMATE,								// number of near neighbors
				caustics_nnIdx,							// nearest neighbors (returned)
				caustics_dists,							// distance (returned)
				0);								// error bound

		// compute SUM(BRDF * PhotonPowers) around pHit
		T caustics_maxDist = 0;

		// find max distance
		for (unsigned i = 0; i < CAUSTICS_ESTIMATE; i++)
		{
			if (caustics_maxDist < caustics_dists[i])
				caustics_maxDist = caustics_dists[i];
		}

		for (unsigned i = 0; i < CAUSTICS_ESTIMATE; i++)
		{
			// Gaussian filter
			//T w = (0.918)*(1 - ((1 - exp(-(1.953)*(pow(dists[i],2)/(2*maxDist))))/(1 - exp(-1.953))));
			
			T w = 1 - caustics_dists[i]/(k*caustics_maxDist);
			//T w = 1;
			sum += causticsMap[caustics_nnIdx[i]].power * w * obj->computeBRDF();
		}

		// solve the approximate rendering function
		// by dividing "sum" by the area of the projected sphere 
		// around pHit.
		// 1 - (2/3k) is the normalization term for the cone filter
		T md = max(maxDist, caustics_maxDist);
		sum /= (T)PI*md*(1-2/(k*3));

		return sum;
	}

	Vec3<T> trace(const Vec3<T> &rayOrig, const Vec3<T> &rayDir, const int depth)
	{
		Vec3<T> pHit;						// point of intersection
		Vec3<T> nHit;						// normal at the intersection point
		GeomObject<T> *obj = NULL;
		if (!checkIntersection(rayOrig, rayDir, obj, pHit, nHit))	// closest object
			return Vec3<T>(0);

		Vec3<T> surfaceColor = 0; // color of the ray/surfaceof the object intersected by the ray

		nHit.normalize(); // normalize normal direction

		// If the normal and the view direction are not opposite to each other 
		// reverse the normal direction. That also means we are inside the sphere so set
		// the inside bool to true. Finally reverse the sign of IdotN which we want
		// positive.
		T bias = 1e-4; // add some bias to the point from which we will be tracing
		bool inside = false;
		if (rayDir.dot(nHit) > 0) nHit = -nHit, inside = true;
		if ((obj->transparency > 0 || obj->reflection > 0) && depth < MAX_RAY_DEPTH)
		{
			T facingratio = -rayDir.dot(nHit);
			// change the mix value to tweak the effect
			T fresneleffect = mix<T>(pow(1 - facingratio, 3), 1, 0.6); 
			// compute reflection direction (not need to normalize because all vectors
			// are already normalized)
			Vec3<T> refldir = rayDir - nHit * 2 * rayDir.dot(nHit);
			refldir.normalize();
			Vec3<T> reflection = trace(pHit + nHit * bias, refldir, depth + 1);

			Vec3<T> refraction = 0;
			// if the sphere is also transparent compute refraction ray (transmission)
			if (obj->transparency) {
				T ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
				T cosi = -nHit.dot(rayDir);
				T k = 1 - eta * eta * (1 - cosi * cosi);
				Vec3<T> refrdir = rayDir * eta + nHit * (eta *  cosi - sqrt(k));
				refrdir.normalize();
				refraction = trace(pHit - nHit * bias, refrdir, depth + 1);
			}
			// the result is a mix of reflection and refraction (if the sphere is transparent)
			surfaceColor = (reflection * fresneleffect * obj->reflection + 
				refraction * (1) * obj->transparency) * obj->surfaceColor;
		}
		else 
			surfaceColor = computeRadiance(pHit, obj);

		return surfaceColor;
	}

public:
	
	// Regular Photon Map
	vector<Photon> photonMap;					// Photon list

	int					nPts;					// actual number of data points
	ANNpointArray		dataPts;				// data points
	
	ANNidxArray			nnIdx;					// near neighbor indices
	ANNdistArray		dists;					// near neighbor distances
	ANNkd_tree*			kdTree;					// KD Tree

	// Caustics Photon Map
	vector<Photon> causticsMap;					// Caustics map
	int					caustics_nPts;					// actual number of data points
	ANNpointArray		caustics_dataPts;				// data points
	
	ANNidxArray			caustics_nnIdx;					// near neighbor indices
	ANNdistArray		caustics_dists;					// near neighbor distances
	ANNkd_tree*			caustics_kdTree;		// KD Tree for caustics

	PhotonMapper()
	{
		nnIdx = new ANNidx[ESTIMATE];						// allocate near neigh indices
		dists = new ANNdist[ESTIMATE];						// allocate near neighbor dists
		nPts = 0;

		caustics_nnIdx = new ANNidx[CAUSTICS_ESTIMATE];
		caustics_dists = new ANNdist[CAUSTICS_ESTIMATE];
		caustics_nPts = 0;
	}

	~PhotonMapper()
	{
		delete[] nnIdx;	
		delete[] dists;
		delete kdTree;

		delete[] caustics_nnIdx;	
		delete[] caustics_dists;
		delete caustics_kdTree;
	}

	//----------------------FIRST-PASS----------------------
	/* The first pass of Photon Mapping algorithm takes care
	to throw all the photons through the scene, inserting them
	in a vector.
	After all the photon are stored the kd-tree is built in 
	order to execute the second pass of the algorithm. */

	// emit photons from the lights
	// assuming power equals for every light
	// assuming diffuse point lights
	void emitPhotons()
	{
		for (unsigned int i = 0; i < g_lights.size(); i++)
		{
			int emittedPhotons = 0;
			while(emittedPhotons < PHOTONS)								// emit "PHOTONS" photons from each light
			{
				Vec3<T> rayDir = randomPointInUnitSphere();				// for each photon compute random direction
				rayDir.normalize();

				Photon p;
				p.power = Vec3<T>((T)LIGHT_POWER / PHOTONS);
				p.power.gate(0,1);

				tracePhoton(g_lights[i]->getPosition(), rayDir, p, 0, false);	// trace photon from the light to the rnd direction

				emittedPhotons++;
			}
		}

		dataPts = annAllocPts(photonMap.size(), 3);			// allocate data points (max num of data points, dimension)

		// load data points
		for (unsigned i = 0; i < photonMap.size(); i++)
		{
			Photon p = photonMap[i];
			dataPts[nPts][0] = p.x;		// load data points for ANN
			dataPts[nPts][1] = p.y;
			dataPts[nPts][2] = p.z;

			nPts++;
		}

		// build kd-tree
		kdTree = new ANNkd_tree(					// build search structure
						dataPts,					// the data points
						nPts,						// number of points
						3);							// dimension of space
	}

	// Emit Photons from the light to reflective/refractive objects
	void emitCausticPhotons()
	{
		for (unsigned int i = 0; i < g_lights.size(); i++)
		{
			for (unsigned int j = 0; j < g_caustics.size(); j++)
			{
				int emittedPhotons = 0;
				while(emittedPhotons < CAUSTICS_PHOTONS)					// emit "CAUSTICS_PHOTONS" photons from each light
				{
					Vec3<T> rayDir = g_caustics[j]->randomPoint() - g_lights[i]->getPosition();			// for each photon compute random direction within the object
					rayDir.normalize();

					Photon p;
					p.power = Vec3<T>((T)LIGHT_POWER / PHOTONS);
					p.power.gate(0,1);

					tracePhoton(g_lights[i]->getPosition(), rayDir, p, 0, true);	// trace photon from the light to the rnd direction

					emittedPhotons++;
				}
			}
			
		}

		caustics_dataPts = annAllocPts(causticsMap.size(), 3);			// allocate data points (max num of data points, dimension)

		// load data points
		for (unsigned i = 0; i < causticsMap.size(); i++)
		{
			Photon p = causticsMap[i];
			caustics_dataPts[caustics_nPts][0] = p.x;		// load data points for ANN
			caustics_dataPts[caustics_nPts][1] = p.y;
			caustics_dataPts[caustics_nPts][2] = p.z;

			caustics_nPts++;
		}

		// build kd-tree
		caustics_kdTree = new ANNkd_tree(					// build search structure
								caustics_dataPts,					// the data points
								caustics_nPts,						// number of points
								3);							// dimension of space
	}
	//----------------------SECOND-PASS----------------------
	/* In the second pass, the algorithm executes standard
	ray-tracing (shooting ray from the camera toward the scene)
	gathering N photons every time a ray intersects an object.
	The gathering phase is performed using Nearest Neighbours
	algorithm which relies on the kd-tree previously defined. */

	// given a ray, this method collects the photons around the 
	// point in which the ray intersects an object, in order 
	// to compute the data needed for the final rendering.
	// The method returns the color of the pixel.
	Vec3<T> gatherPhotons(const Vec3<T> &rayOrig, const Vec3<T> &rayDir)
	{
		return trace(rayOrig, rayDir, 0) * EXPOSURE;
	}
	
};

//------------------------------------
//--------------Renderer--------------
//------------------------------------

//Renders using QT
class Renderer {

public:
	Renderer()
	{
		g_window = new QWidget();
		g_image = new QImage(WINDOW_WIDTH, WINDOW_HEIGHT, QImage::Format_RGB32);
		g_label = new QLabel(g_window);
	}

	~Renderer()
	{
		
	}

	void render()
	{
		g_label->setPixmap(QPixmap::fromImage(*g_image));
		g_label->show();
		g_window->show();
	}
};

//-----------------------------------------------
//--------------Program Entry Point--------------
//-----------------------------------------------
int main(int argc, char *argv[])
{
	// init QT application
	QApplication app(argc, argv);

	// set camera: position, look at, up vector, right vector
	Camera<float>* camera = new Camera<float>(Vec3<float>(0,0,-30), g_origin, Vec3<float>(0,1,0), Vec3<float>(1,0,0));
	
	// init renderer
	Renderer* renderer = new Renderer();

	// init RayTracer
	RayTracer<float>* rayTracer = new RayTracer<float>();

	// init Photon Mapper
	PhotonMapper<float>* photonMapper = new PhotonMapper<float>();

	// push Diffuse spheres
	// position, radius, surface color, reflectivity, transparency, emission color
	g_objects.push_back(new Sphere<float>(Vec3<float>(-10, -11, 0), 8, (Vec3<float>(255, 255, 255))/255, 1, 1));

	// Caustics objects
	GeomObject<float>* s2 = new Sphere<float>(Vec3<float>(0, 0, 0), 8, (Vec3<float>(67, 182, 240))/255, 1, 1);
	//g_objects.push_back(s2);
	//g_caustics.push_back(s2);

	// push planes
	// position, normal, surface color, reflectivity, transparency, emission color 
	g_objects.push_back(new Plane<float>(Vec3<float>(-20, 0, 0), Vec3<float>(1, 0, 0), Vec3<float>(237, 21, 21)/255, 1, 0));
	g_objects.push_back(new Plane<float>(Vec3<float>(0, 0, 20), Vec3<float>(0, 0, -1), Vec3<float>(14, 237, 48)/255, 0, 0));
	g_objects.push_back(new Plane<float>(Vec3<float>(20, 0, 0), Vec3<float>(-1, 0, 0), Vec3<float>(40, 145, 250)/255, 1, 0));
	g_objects.push_back(new Plane<float>(Vec3<float>(0, 20, 0), Vec3<float>(0, -1, 0), Vec3<float>(1, 1, 1), 0, 0));
	g_objects.push_back(new Plane<float>(Vec3<float>(0, -20, 0), Vec3<float>(0, 1, 0), Vec3<float>(1, 1, 1), 1, 0));

	// push lights
	Sphere<float>* light_1 = new Sphere<float>(Vec3<float>(0, 17, 0), 0.3, Vec3<float>(1), 0, 0, Vec3<float>(1));
	g_lights.push_back(light_1);

	// Seed the random generator
	int seed = static_cast<int>(time(0));
	srand(seed);

	#ifdef PHOTON_MAPPING

	// Photon Mapping
	photonMapper->emitPhotons();
	photonMapper->emitCausticPhotons();
	
	for (int i = 0; i < WINDOW_WIDTH; i++)
	{
		for (int j = 0; j < WINDOW_HEIGHT; j++)
		{
			Vec3<float> rayDir = camera->pixelToWorld(i,j,0.5f,0.5f) - camera->position;
			Vec3<float> px_color = photonMapper->gatherPhotons(camera->position, rayDir) * 255; //remap to [0,255]
			
			px_color.gate(0,255);	// prevent overflow
			g_image->setPixel(i,j,qRgb(px_color.x, px_color.y, px_color.z));
		}
	}

	#else

	// RayTracing
	for (int i = 0; i < WINDOW_WIDTH; i++)
	{
		for (int j = 0; j < WINDOW_HEIGHT; j++)
		{
			Vec3<float> rayDir = camera->pixelToWorld(i,j,0.5f,0.5f) - camera->position;
			Vec3<float> px_color = rayTracer->trace(camera->position, rayDir, 0) * 255; //remap to [0,255]
			
			g_image->setPixel(i,j,qRgb(px_color.x, px_color.y, px_color.z));
		}
	}

	#endif	//PHOTON_MAPPING

	// render
	renderer->render();
	app.exec();
	
	// clean things up
	delete renderer;
	delete photonMapper;
	annClose();							// done with ANN

	// free global variables
	/*delete g_window;
	delete g_image;
	delete g_label;*/
	
	return 0;
}